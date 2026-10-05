"""CPU regression tests: python -m unittest discover -s tests -v"""
import tempfile
import json
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset
from code import config as C
from code.model.BiAG import BiAGWrapper, SCM
from code.model.classifier import CosineClassifier
from code.engine.train_biag import _random_episode, classwise_cosine, run
from code.engine.incremental_eval import evaluate, _classification_metrics
from code.data.data_utils import CutMixCollate, get_new_dataloader
from code.data.cifar100.cifar100 import CIFAR100
from code.data.base_loader import build_fscil_loaders
from code.utils.utils import seed_everything
from main import build_main_parser


class BiAGTests(unittest.TestCase):
    def setUp(self):
        seed_everything(17)
        torch.set_num_threads(2)
        self.state = SimpleNamespace(
            protos=F.normalize(torch.randn(60, 16), dim=-1),
            weights=[F.normalize(torch.randn(60, 16), dim=-1)], device='cpu')

    def test_episode_and_per_class_loss(self):
        p, gt, old_p, old_w = _random_episode(self.state, 60)
        self.assertEqual(p.shape, (1, 5, 16))
        self.assertEqual(old_p.shape, (1, 55, 16))
        pred = BiAGWrapper(16)(p, old_p, old_w)
        self.assertEqual(classwise_cosine(pred, gt).shape, (5,))
        with self.assertRaises(ValueError):
            classwise_cosine(pred.unsqueeze(1), gt)

    def test_exact_class_assignment_beats_collapse(self):
        gt = torch.eye(5)
        correct = 1 - classwise_cosine(gt, gt).mean()
        collapsed = 1 - classwise_cosine(torch.ones_like(gt), gt).mean()
        permuted = 1 - classwise_cosine(gt.roll(1, 0), gt).mean()
        self.assertEqual(correct.item(), 0.0)
        self.assertGreater(collapsed.item(), correct.item())
        self.assertGreater(permuted.item(), correct.item())

    def test_every_reasoning_block_receives_gradients(self):
        model = BiAGWrapper(16)
        p, gt, old_p, old_w = _random_episode(self.state, 60)
        (1 - classwise_cosine(model(p, old_p, old_w), gt).mean()).backward()
        for block in model.biag.blocks:
            self.assertGreater(sum(x.grad.abs().sum().item() for x in block.parameters()
                                   if x.grad is not None), 0.0)
            self.assertGreater(block.wsa.attn.in_proj_weight.grad[:32].abs().sum().item(), 0.0)
        self.assertEqual(sum(isinstance(m, SCM) for m in model.modules()), 1)

    def test_single_new_class_and_invalid_batch(self):
        model = BiAGWrapper(16)
        p, _, old_p, old_w = _random_episode(self.state, 60, k_way=1)
        self.assertEqual(model(p, old_p, old_w).shape, (1, 16))
        with self.assertRaises(ValueError):
            model(p.expand(5, -1, -1), old_p, old_w)

    def test_training_and_checkpoint_roundtrip(self):
        model = BiAGWrapper(16, depth=2)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=2)
        best, last, cosine = run(self.state, model, 1, 2, optimizer, scheduler)
        restored = BiAGWrapper(16, depth=2)
        restored.load_state_dict(last)
        self.assertTrue(np.isfinite(cosine))
        p, _, op, ow = _random_episode(self.state, 60)
        torch.testing.assert_close(model(p, op, ow), restored(p, op, ow))
        self.assertIn('biag.scm.mlp.0.weight', best)


class EvaluationTests(unittest.TestCase):
    def test_unseen_logits_cannot_win(self):
        loader = [(torch.tensor([[-2., 100., -1.], [-1., 100., -2.]]), torch.tensor([2, 0]))]
        metrics = _classification_metrics(nn.Identity(), loader, 'cpu', {0, 2}, {0})
        self.assertEqual(metrics, dict(acc=100., base_acc=100., novel_acc=100.))

    def test_only_seen_knowledge_is_passed_to_generator(self):
        class Recorder(nn.Module):
            def __init__(self):
                super().__init__()
                self.old_counts = []
            def forward(self, new, old, weights):
                self.old_counts.append(old.shape[1])
                torch.testing.assert_close(old, weights)
                return new[0]
        eye = torch.eye(4)
        clf = CosineClassifier(4, 4)
        with torch.no_grad():
            clf.weight.zero_()
            clf.weight[[0, 2]] = eye[[0, 2]]
        protos = clf.weight.detach().clone()
        protos[[1, 3]] = 999  # Future slots must never enter old knowledge.
        state = SimpleNamespace(device='cpu', backbone=nn.Identity(), classifier=clf,
                                protos=protos, biag=Recorder(), weights=[clf.weight.data])
        state.extract_proto = lambda imgs: F.normalize(imgs.mean(0, keepdim=True), dim=1)
        def batch(ids, repeats=1):
            labels = torch.tensor(ids).repeat_interleave(repeats)
            return [(eye[labels], labels)]
        loaders = dict(d0_test=batch([0, 2]), support_sess=[batch([3], C.SHOT), batch([1], C.SHOT)],
                       test_sess=[batch([0, 2, 3]), batch([0, 2, 3, 1])],
                       class_splits=dict(base=[0, 2], sessions=[[0, 2, 3], [0, 2, 3, 1]]))
        result = evaluate(loaders, state)
        self.assertEqual(state.biag.old_counts, [2, 3])
        self.assertEqual(result['acc_sessions'], [100., 100., 100.])
        self.assertEqual(result['base_acc_sessions'], [100., 100., 100.])
        self.assertEqual(result['novel_acc_sessions'], [None, 100., 100.])
        self.assertEqual(result['forgetting'], [0., 0.])


class DataTests(unittest.TestCase):
    def test_cutmix_labels_match_clipped_pixel_area(self):
        batch = [(torch.zeros(3, 32, 32), 0), (torch.ones(3, 32, 32), 1)]
        with patch('numpy.random.beta', return_value=.5), patch('numpy.random.randint', return_value=0), \
                patch('torch.randperm', return_value=torch.tensor([1, 0])):
            imgs, (_, _, lam) = CutMixCollate(prob=1.)(batch)
        self.assertAlmostEqual(lam, 1 - 121 / 1024)
        self.assertAlmostEqual(imgs[0].mean().item(), 1 - lam)

    def test_cutmix_respects_seed(self):
        batch = [(torch.zeros(3, 32, 32), 0), (torch.ones(3, 32, 32), 1)]
        seed_everything(4)
        a, ta = CutMixCollate(prob=1.)(batch)
        seed_everything(4)
        b, tb = CutMixCollate(prob=1.)(batch)
        torch.testing.assert_close(a, b)
        self.assertEqual(ta[2], tb[2])

    def test_cifar_support_is_plain_and_uses_training_split(self):
        pixels = np.arange(3072, dtype=np.uint8)[None, :]
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp) / CIFAR100.base_folder
            folder.mkdir()
            (folder / 'train').touch()
            with patch.object(CIFAR100, '_check_integrity', return_value=True), \
                    patch.object(CIFAR100, '_load_meta'), \
                    patch.object(CIFAR100, 'NewClassSelector', lambda self, data, targets, index: (data, targets)), \
                    patch('code.data.cifar100.cifar100.pickle.load', return_value=dict(data=pixels, fine_labels=[60])):
                dataset = CIFAR100(tmp, train=True, index=[0], base_sess=False, do_augment=False)
            self.assertEqual([type(t).__name__ for t in dataset.transform.transforms], ['ToTensor', 'Normalize'])
            torch.testing.assert_close(dataset[0][0], dataset[0][0])
            self.assertEqual(dataset[0][1], 60)

    def test_incremental_loader_forwards_no_augmentation(self):
        calls = []
        class FakeDataset(TensorDataset):
            def __init__(self, **kwargs):
                calls.append(kwargs)
                super().__init__(torch.zeros(25, 3, 32, 32), torch.arange(25))
        args = SimpleNamespace(dataset='cifar100', Dataset=SimpleNamespace(CIFAR100=FakeDataset),
                               data_folder='unused', num_workers=0, batch_size_inference=32,
                               base_class=60, way=5)
        get_new_dataloader(args, 1, do_augment=False)
        self.assertIs(calls[0]['do_augment'], False)
        self.assertEqual(len(calls[0]['index']), 25)
        self.assertEqual(list(calls[1]['index']), list(range(65)))

    def test_base_loaders_do_not_load_incremental_support(self):
        calls = []
        class FakeDataset(TensorDataset):
            def __init__(self, **kwargs):
                calls.append(kwargs)
                self.targets = list(kwargs['index'])
                super().__init__(torch.zeros(60, 3, 84, 84), torch.arange(60))
        with patch('code.data.base_loader.MiniImageNet', FakeDataset), patch.object(C, 'NUM_WORKERS', 0):
            loaders = build_fscil_loaders('miniimagenet')
        self.assertEqual(len(calls), 3)
        self.assertTrue(all(c['base_sess'] for c in calls))
        self.assertEqual(loaders['d0_test'].dataset.targets, list(range(60)))


class ConfigTests(unittest.TestCase):
    def test_cli_dataset_and_depth_are_effective(self):
        snapshot = C.effective_config_dict()
        try:
            args = build_main_parser().parse_args(['incremental_run', '--dataset', 'miniimagenet', '--biag_depth', '2'])
            C.update_from_args(vars(args))
            self.assertEqual((C.IMAGE_SIZE, C.BACKBONE_MODEL, C.SESSIONS), (84, 'resnet12', 9))
            self.assertEqual(len(BiAGWrapper(16).biag.blocks), 2)
            C.update_from_args(dict(dataset='miniimagenet', backbone_model='resnet18'))
            self.assertEqual(C.BACKBONE_MODEL, 'resnet18')
        finally:
            for key, value in snapshot.items():
                setattr(C, key, value)


class PipelineTests(unittest.TestCase):
    def test_three_cli_stages_and_legacy_checkpoint_rejection(self):
        from scripts import base, biag, incremental_run
        from code.engine.incremental_eval import load_state
        class TinyBackbone(nn.Module):
            out_dim = 16
            def __init__(self):
                super().__init__()
                self.proj = nn.Linear(16, 16)
            def forward(self, x):
                return F.normalize(self.proj(x), dim=-1)
        seed_everything(10)
        features = torch.randn(65, 16)
        base_batch = [(features[:60], torch.arange(60))]
        base_loaders = dict(d0_train=base_batch, proto=base_batch, d0_test=base_batch,
                            class_splits=dict(base=list(range(60))))
        new_labels = torch.arange(60, 65).repeat_interleave(5)
        eval_loaders = dict(d0_test=base_batch,
                            support_sess=[[(features[new_labels], new_labels)]],
                            test_sess=[[(features, torch.arange(65))]],
                            class_splits=dict(base=list(range(60)), sessions=[list(range(65))]))
        snapshot = C.effective_config_dict()
        try:
            with tempfile.TemporaryDirectory() as tmp, \
                    patch('scripts.base.ResNet18', TinyBackbone), \
                    patch('scripts.biag.ResNet18', TinyBackbone), \
                    patch('code.engine.incremental_eval.ResNet18', TinyBackbone), \
                    patch('scripts.base.build_fscil_loaders', return_value=base_loaders), \
                    patch('scripts.incremental_run.build_cifar_fscil_loaders', return_value=eval_loaders), \
                    patch.object(C, 'PSEUDO_EPISODES_PER_EPOCH', 2):
                args = build_main_parser().parse_args(['all', '--dataset', 'cifar100',
                    '--output_dir', tmp, '--epochs', '1', '--biag_epochs', '1', '--biag_depth', '2'])
                base.main(args)
                biag.main(args)
                incremental_run.main(args)
                output = Path(tmp) / 'cifar100'
                summary = json.loads((output / 'run/summary.json').read_text())
                self.assertEqual(len(summary['acc_sessions']), 2)
                self.assertIsNotNone(summary['novel_acc_sessions'][1])
                loaded = load_state(args)
                saved_classifier = torch.load(output / 'classifier_pt_last.pt', weights_only=True)
                torch.testing.assert_close(loaded.classifier.scale.cpu(), saved_classifier['scale'].cpu())
                torch.save({'biag.blocks.0.scm.mlp.0.weight': torch.ones(1)}, output / 'biag_pt_last.pt')
                with self.assertRaisesRegex(ValueError, 'Legacy BiAG checkpoint'):
                    load_state(args)
        finally:
            for key, value in snapshot.items():
                setattr(C, key, value)


if __name__ == '__main__':
    unittest.main()
