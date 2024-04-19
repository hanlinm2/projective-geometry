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