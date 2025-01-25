# ncnnPicVec
基于ncnn和MobileCLIP（ViT部分）的图片向量聚类

本项目是帮友人的表情包相关的一款APP写的，自己没其他地方用，因此开源。

得益于 MobileCLIP（模型权重大小不到25MB）强大的泛化性，本项目能实现 Zero Shot分类（例如Demo，数据库里全是MyGO的角色A，但是输入角色B能识别出这是MyGO的角色）。
本项目的模型取自Apple的MobileCLIP-s0，由于TextEncoder部分较大（80mb），且不支持中文，私以为没有落地的价值，因此没有落地，如有需求请发ISSUES，足够多我会考虑。
