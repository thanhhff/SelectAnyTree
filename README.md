# 🌲 SelectAnyTree
### A Promptable Instance Segmentation Model for 3D Forest LiDAR Point Clouds

![Status](https://img.shields.io/badge/status-pre--release-orange)
[![Paper](https://img.shields.io/badge/paper-arXiv%3A2606.27491-b31b1b)](https://arxiv.org/abs/2606.27491)
![Code](https://img.shields.io/badge/code-coming%20soon-lightgrey)
![License](https://img.shields.io/badge/license-CC%20BY--NC%204.0-blue)

---

**SelectAnyTree** is a *promptable* instance segmentation model for 3D forest LiDAR
point clouds. A user clicks a point → that individual tree is segmented;
positive/negative clicks then refine the mask. The scene is encoded **once** and
reused across all prompts, so many trees can be selected interactively without
recomputing features.

> [!NOTE]
> 📄 **The paper is now available on arXiv:** [2606.27491](https://arxiv.org/abs/2606.27491).
> 🚧 Source code, pretrained checkpoint, and the interactive web demo are still
> coming soon. ⭐ Star and watch this repo to be notified.

## 🧠 Abstract

Automated instance segmentation of forest LiDAR point clouds is increasingly critical as forest monitoring moves toward scalable, detailed, 3D measurement, but progress is constrained by label scarcity for tree instances — a single hectare can hold millions of points and hundreds of overlapping crowns, making manual annotation laborious and error-prone. Existing pre-segmentation tools offer no interactive or AI-assisted refinement. Inspired by promptable foundation segmentation models, SelectAnyTree delineates any individual tree in a 3D forest point cloud from a few clicks, converting each click into a decoder query that fuses its 3D position, polarity, and local backbone feature, and using the CHM treetop as a geometry-guided prompt. Evaluated across seven diverse forest regions plus an independent held-out test set, it segments a target tree to **78.2 IoU from a single click** — 24.8 points above the strongest promptable baseline — reaching every accuracy target with the fewest clicks, while using far fewer parameters and less inference time than prior promptable models.

## 📦 Release checklist

- 🌲 **[2026/06]** Paper
- [ ] Source code
- [ ] Pretrained checkpoint
- [ ] Interactive web demo

## 📚 Citation

If you find SelectAnyTree useful in your research, please consider citing:

```bibtex
@article{nguyen2026selectanytree,
  title   = {SelectAnyTree: A Promptable Instance Segmentation Model for 3D Forest LiDAR Point Clouds},
  author  = {Nguyen, Trung Thanh and Lusk, Daniel and Gerberding, Kilian and
             Vajna-Jehle, Janusch and Vu, Tuan-Anh and Le, Duc Viet and Vo, Tu and
             Nguyen, Phi Le and Kawanishi, Yasutomo and Komamizu, Takahiro and
             Ide, Ichiro and Frey, Julian and Kattenborn, Teja},
  journal = {arXiv preprint arXiv:2606.27491},
  year    = {2026}
}
```

## 📄 License

SelectAnyTree is built on the ForestMamba / ForestFormer3D codebase, which is based
on [OneFormer3D](https://github.com/filaPro/oneformer3d) by Danila Rukhovich,
licensed under CC BY-NC 4.0. This repository will be released under the same license.
