# *Shadows Don’t Lie and Lines Can’t Bend!* <br> Generative Models don’t know Projective Geometry...for now

Ayush Sarkar*,
[Hanlin Mai*](https://hanlinmai.web.illinois.edu),
Amitabh Mahapatra*,
[Svetlana Lazebnik](https://slazebni.cs.illinois.edu/),
[David A. Forsyth](http://luthuli.cs.uiuc.edu/~daf/),
[Anand Bhattad&dagger;](https://anandbhattad.github.io/)

University of Illinois Urbana-Champaign, &dagger;Toyota Technological Institute at Chicago

*equal contribution

Abstract: *Generative models can produce impressively realistic images. This paper demonstrates that generated images have geometric features different from those of real images.  We  build a set of collections of generated images, prequalified to fool simple, signal-based classifiers into believing they are real.  We then show that prequalified generated images can be identified reliably by classifiers that only look at geometric properties.  We use three such classifiers.  All three classifiers are denied access to image pixels, and look only at derived geometric features. The first classifier looks at the perspective field of th image, the second looks at lines detected in the image, and the third looks at relations between detected objects and shadows.  Our procedure detects generated images more reliably than SOTA local signal based detectors, for images from a number of distinct generators. Saliency maps suggest that the classifiers can identify geometric problems reliably. We conclude that current generators cannot reliably reproduce geometric properties of real images.*

<a href="https://arxiv.org/abs/2311.17138"><img src="https://img.shields.io/badge/arXiv-2311.17138-b31b1b.svg" height=22.5></a>
<a href="https://projective-geometry.github.io/"><img src="https://img.shields.io/website?down_color=lightgrey&down_message=offline&label=Project%20Page&up_color=lightgreen&up_message=online&url=https%3A%2F%2Fprojective-geometry.github.io" height=22.5></a>

<p align="center">
<img max-height="200px" alt="Teaser Figure" src="./assets/teaser.png">
</p>

# Getting Started

## Dataset Download

Our experiments are done on the dataset with 256 x 256 resolution, which contains generated and real images with a train/val/test split. 

We also release a high quality (1024 x 1024) dataset that contains only generated images for future research. 

Download the dataset and put them in the `./dataset` folder 

| Dataset (256x256) | High Quality Dataset (1024x1024) |
| ------------- | ------------- |
| [Download](https://huggingface.co/datasets/amitabh3/Projective-Geometry) | [Download](https://huggingface.co/datasets/amitabh3/Projective-Geometry-1024)  |

## Training and Testing

Navigate to one of the following directories and follow the instructions in the README

`prequalifier/`  
`object_shadow/`   
`perspective_fields/`  
`line_segment/`  

## BibTex

```
@InProceedings{Sarkar_2024_CVPR,
    author    = {Sarkar, Ayush and Mai, Hanlin and Mahapatra, Amitabh and Lazebnik, Svetlana and Forsyth, D.A. and Bhattad, Anand},
    title     = {Shadows Don't Lie and Lines Can't Bend! Generative Models don't know Projective Geometry...for now},
    booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
    month     = {June},
    year      = {2024},
    pages     = {28140-28149}
}
```