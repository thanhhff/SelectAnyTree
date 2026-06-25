# 🌲 SelectAnyTree

### A Promptable Instance Segmentation Model for 3D Forest LiDAR Point Clouds

![Status](https://img.shields.io/badge/status-pre--release-orange)
![Paper](https://img.shields.io/badge/paper-coming%20soon-lightgrey)
![Code](https://img.shields.io/badge/code-coming%20soon-lightgrey)
![License](https://img.shields.io/badge/license-CC%20BY--NC%204.0-blue)

---

**SelectAnyTree** is a *promptable* instance segmentation model for 3D forest LiDAR
point clouds. A user clicks a point → that individual tree is segmented;
positive/negative clicks then refine the mask. The scene is encoded **once** and
reused across all prompts, so many trees can be selected interactively without
recomputing features.

> [!NOTE]
> 🚧 **This is a pre-release.** The source code, pretrained checkpoint, and
> interactive web demo will be released soon. ⭐ Star and watch this repo to be
> notified.

## 📦 Release checklist

- [ ] Source code
- [ ] Pretrained checkpoint
- [ ] Interactive web demo
- [ ] Paper

## 📚 Citation

If you find SelectAnyTree useful in your research, please consider citing:

```bibtex
@article{nguyen2026selectanytree,
  title   = {SelectAnyTree: A Promptable Instance Segmentation Model for 3D Forest LiDAR Point Clouds},
  author  = {Nguyen, Trung Thanh and Lusk, Daniel and Gerberding, Kilian and
             Vajna-Jehle, Janusch and Vu, Tuan-Anh and Le, Duc Viet and Vo, Tu and
             Nguyen, Phi Le and Kawanishi, Yasutomo and Komamizu, Takahiro and
             Ide, Ichiro and Frey, Julian and Kattenborn, Teja},
  year    = {2026}
}
```

## 📄 License

SelectAnyTree is built on the ForestMamba / ForestFormer3D codebase, which is based
on [OneFormer3D](https://github.com/filaPro/oneformer3d) by Danila Rukhovich,
licensed under CC BY-NC 4.0. This repository will be released under the same license.
