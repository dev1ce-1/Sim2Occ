# Sim2Occ

**Selective feature transfer from simulated dense traffic scenes for 3D semantic occupancy prediction.** Sim2Occ matches observed traffic layouts to CARLA roads and assets. During training, retrieval-guided partial alignment (RPA, called MatchMass-Occ in the experiments) matches local real features to same-class simulated regions. The fraction of matched mass sets the strength of the auxiliary target.

| Module | Implementation |
| --- | --- |
| [Roads](sim2occ/roads.py) and [assets](sim2occ/assets.py) | Rank road and sidewalk layouts with distance-weighted Dice, then rank same-class assets by their fit across camera views. |
| [Occupancy labels](sim2occ/labels.py) | Resolve measured surface and free-space evidence. Ambiguous endpoints remain unknown. |
| [Tokens](sim2occ/tokens.py) and [alignment](sim2occ/transport.py) | Construct local descriptors, retrieve candidates, solve partial transport and compute the transfer loss. |
| [Model adapters](sim2occ/adapters.py) | Sample dense occupancy features on a canonical voxel grid and cap the auxiliary gradient relative to the task gradient. |
| [Comparison methods](sim2occ/baselines.py) and [metrics](sim2occ/metrics.py) | CCSA, Deep CORAL, FDA and confusion-matrix scoring for 17 occupied semantic classes. |

## Run a CPU example

From the repository root, install the dependencies with `python -m pip install -r requirements.txt`. This example aligns one real token with a synthetic memory entry and backpropagates the auxiliary loss.

```python
import torch
from sim2occ.protocol import RPA
from sim2occ.tokens import TokenRecord, select_descriptors
from sim2occ.transport import align_queries, transfer_loss

labels = torch.full((8, 8, 4), 17)  # free space
labels[3, 3, 1] = 4                 # one occupied car voxel
visible = torch.ones_like(labels, dtype=torch.bool)
source, _ = select_descriptors(labels, visible, "sim", "scene_s", seed=0)
query, _ = select_descriptors(labels, visible, "real", "scene_r", seed=0)
student = torch.tensor([[1., 0.]], requires_grad=True)
bank = [TokenRecord(source[0], torch.tensor([1., 0.]), torch.tensor([0.8, 0.6]))]
target, mass, keep, _ = align_queries(RPA, query, student.detach(), bank, seed=0, event=0)
loss = transfer_loss(student, target, mass, keep, n_all=len(query))
loss.backward()
print(f"accepted={int(keep.sum())}, matched mass={mass.item():.4f}, loss={loss.item():.4f}")
```

For the FlashOCC experiments, the source bank used 888 synthetic frames, up to 64 records per class and 1,088 overall. The default ran retrieval every four updates with 16 same-class candidates, geometry weight 0.5, dustbin cost 0.5, transfer coefficient `0.5 × matched mass` and an auxiliary-gradient cap of 0.1. The original 888 captures used layouts from 29 validation scenes. Account for this overlap when evaluating unseen-scene generalization.

This lightweight release contains the method components. Training environments, CARLA assets, nuScenes data and model weights are supplied separately by their respective providers.
