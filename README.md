# MALMO

## Table of Contents
Introduction
Our Study
Our Pipeline
Demo Video
Research Output
--


## Introduction
The MALMO [***Mathematical approaches to modelling metabolic plasticity and heterogeneity in Melanoma***] Project aims at understanding how tumor heterogeneity contributes to melanoma progression and treatment resistance. The infiltration of cancer cells within the host tissue cause molecular, cellular, and physical changes that lead to the creation of a tumor microenvironment. Such environments are deprived of nutrients and oxygen typically available in normally functioning tissue. Despite this, melanoma cells have the ability to adapt and rewire themselves under these changing conditions, and thus continue to grow and proliferate, which makes this cancer a challenge to treat. 

## Our Study
Our assumption is that treatment resistance can be explained by capturing these non-genetic transitions and changing metabolic states. Our study has utilized artificial intelligence (AI) and mechanistic modeling to capture these evolving states, with a preliminary focus on the blood vessel as a biomarker, given vessels are both providers of nutrients and oxygen, as well as facilitators of waste management within the body. 

## Our Pipeline
We have developed a 2D- and 3D-based pipeline for evaluation of whole slide images (WSI) - digitized pathology slides - with particular focus on hematoxylin and eosin (H&E) and cluster of differentiation 31 (CD31) stained tissue sections. Tissue sections were acquired using Patient-Derived Xenograpft (PDX) mouse models. In the pipeline, images first undergo preprocessing including (A) being exported in Tagged Image File (TIF) Format, (B) split into top and bottom tissues, and (C) crop-centered, inpainted, and cleaned via the removal of the noisy background. Once images are pre-processed, they are (D) registered, then undergo patch-level segmentation (E & F) to produce complete, segmented WSIs (G). All segmented WSIs are then rendered and interpolated to produce a final 3D vasculature model (H & I). 

<p align="center">
  <img width="95%" src="/images/3D_WSI_Pipeline_Figure.png">
 </p>

## Demo Video
<div align="center">
  <video src="https://github.com/janan-arslan/MALMO/assets/95415605/1a1226da-c33f-41de-99b8-97722f76bf47">
</div>


## Research Output
[Reconstruction vasculaire 3D et analyse de lames virtuelles H&E dans l'étude du mélanome](https://hal.science/hal-03928851). *IABM2023 : Colloque Français d'Intelligence Artificielle en Imagerie Biomédicale*, Mar 2023, Paris, France.

[Efficient 3D Reconstruction of H&E Whole Slide Images in Melanoma](https://hal.science/hal-03834014). *SPIE Medical Imaging 2023*, Feb 2023, San Diego, California, United States.

[3D reconstruction and mathematical modelling of whole slide images to elucidate resistance to the targeted therapy in melanoma](https://hal.science/hal-03814995). *International Conference in Systems Biology*, Oct 2022, Berlin, Germany.

[Introducing [MALMO]: Mathematical approaches to modelling metabolic plasticity and heterogeneity in Melanoma](https://hal.science/hal-03834055). *RITS 2022 - Recherche en Imagerie et Technologie pour la Santé*, May 2022, Brest, France.

[Data driven mechanistic modeling of oxygen distribution and hypoxia profile in tumor microenvironment](https://hal.science/hal-03834400). *COMPSYSCAN2022: A complex systems approach to cancer understanding*, Oct 2022, Lyon, France.

[Estimating spatial distribution of oxygen and hypoxia in tumor microenvironment: a mechanistic approach](https://hal.science/hal-04021831). *CanceropoleGSO 2022*, Nov 2022, Montpellier, France.





