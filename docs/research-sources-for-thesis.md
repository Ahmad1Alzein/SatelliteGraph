# SatelliteGraph research and source inventory

**Checked 5 October 2026.** This is a citation ledger for a future thesis or
article. It records the papers and primary technical sources traceable in the
project discussions, repository notes, and implementation review. “Consulted”
means the publication record, abstract, or accessible paper was examined for
the stated methodological point; it does **not** assert that every full text
was read end to end. Do not copy a paper's benchmark accuracy into a claim
about this project. The project-specific thresholds and edge labels are our
engineering decisions unless explicitly stated otherwise.

The professor-facing explanation of the implemented rules is
[How two-year changes are identified](temporal-edge-methodology-for-review.md).
The stage documentation gives the exact implementation and artifact contracts.

## Papers directly connected to implemented components

| ID | Thesis-ready reference and primary link | Role in this project |
|---|---|---|
| P01 | Ravi, N., et al. (2024). [*SAM 2: Segment Anything in Images and Videos*](https://arxiv.org/abs/2408.00714). arXiv:2408.00714. | The SAM 2.1 model supplies class-agnostic segment proposals in Stage 2. This project uses pretrained inference, not the paper's video-training method. |
| P02 | Xia, J., Yokoya, N., Adriano, B., & Broni-Bediako, C. (2023). [*OpenEarthMap: A Benchmark Dataset for Global High-Resolution Land Cover Mapping*](https://openaccess.thecvf.com/content/WACV2023/html/Xia_OpenEarthMap_A_Benchmark_Dataset_for_Global_High-Resolution_Land_Cover_Mapping_WACV_2023_paper.html). WACV, 6254–6264. | Eight known land-cover classes and the model/data context for Stage 2 semantic probabilities. The ninth `unknown` class is added by this project. |
| P03 | Chen, W., Gong, X., Liu, X., Zhang, Q., Li, Y., & Wang, Z. (2020). [*FasterSeg: Searching for Faster Real-Time Semantic Segmentation*](https://openreview.net/pdf?id=BJgqQ6NYvB). ICLR. | Architecture lineage for the OpenEarthMap FasterSeg model bundled with the project; the model weights are separately supplied assets. |
| P04 | Huang, B., Reichman, D., Collins, L. M., Bradbury, K., & Malof, J. M. (2018). [*Tiling and Stitching Segmentation Output for Remote Sensing: Basic Challenges and Recommendations*](https://arxiv.org/abs/1805.12219). arXiv:1805.12219. | Motivates overlap reconciliation when Stage 4 combines tile-level predictions. The project's exact ownership rule is its own implementation. |
| P05 | Akyon, F. C., Altinuc, S. O., & Temizel, A. (2022). [*Slicing Aided Hyper Inference and Fine-tuning for Small Object Detection*](https://arxiv.org/abs/2202.06934). arXiv:2202.06934. | Reference for sliced-image overlap handling. The project does not run the SAHI inference package. |
| P06 | Kirillov, A., He, K., Girshick, R., Rother, C., & Dollár, P. (2019). [*Panoptic Segmentation*](https://openaccess.thecvf.com/content_CVPR_2019/html/Kirillov_Panoptic_Segmentation_CVPR_2019_paper.html). CVPR, 9404–9413. | Conceptual reference for reconciling unique object/region identities; no panoptic network is trained here. |
| P07 | Geifman, Y., & El-Yaniv, R. (2017). [*Selective Classification for Deep Neural Networks*](https://arxiv.org/abs/1705.08500). arXiv:1705.08500. | Motivates abstention in uncertain refinement. The project's deterministic gates do not reproduce a calibrated selective-risk guarantee. |
| P08 | Kampffmeyer, M., Salberg, A.-B., & Jenssen, R. (2016). [*Semantic Segmentation of Small Objects and Modeling of Uncertainty in Urban Remote Sensing Images Using Deep Convolutional Neural Networks*](https://openaccess.thecvf.com/content_cvpr_2016_workshops/w19/html/Kampffmeyer_Semantic_Segmentation_of_CVPR_2016_paper.html). CVPR Workshops. | Supports caution around small objects and uncertainty. It does not prescribe the project's 40-pixel unknown threshold. |
| P09 | Lowe, D. G. (2004). [*Distinctive Image Features from Scale-Invariant Keypoints*](https://doi.org/10.1023/B:VISI.0000029664.99615.94). *International Journal of Computer Vision*, 60, 91–110. | Algorithm lineage for SIFT landmarks used through OpenCV during optional cross-year registration. |
| P10 | Fischler, M. A., & Bolles, R. C. (1981). [*Random Sample Consensus: A Paradigm for Model Fitting with Applications to Image Analysis and Automated Cartography*](https://doi.org/10.1145/358669.358692). *Communications of the ACM*, 24(6), 381–395. | Algorithm lineage for robust affine fitting from registration control points. |
| P11 | Chen, G., Zhao, K., & Powers, R. (2014). [*Assessment of the Image Misregistration Effects on Object-Based Change Detection*](https://pages.charlotte.edu/gang-chen/wp-content/uploads/sites/184/2014/01/Chen_2014_ISPRS_Misregistration-OBCD.pdf). *ISPRS Journal of Photogrammetry and Remote Sensing*, 87, 19–27. [DOI:10.1016/j.isprsjprs.2013.10.007](https://doi.org/10.1016/j.isprsjprs.2013.10.007). | Explains why residual shifts can create false object-level changes; motivates registration audits and boundary tolerance. |
| P12 | Yu, W., Zhou, W., Qian, Y., & Yan, J. (2016). [*A New Approach for Land Cover Classification and Change Analysis: Integrating Backdating and an Object-Based Method*](https://doi.org/10.1016/j.rse.2016.02.030). *Remote Sensing of Environment*, 177, 37–47. | Object-based multi-date land-cover analysis; conceptual basis for comparing class polygons over time. The backdating method is not implemented. |
| P13 | Ma, L., Li, M., Blaschke, T., Ma, X., Tiede, D., Cheng, L., Chen, Z., & Chen, D. (2016). [*Object-Based Change Detection in Urban Areas: The Effects of Segmentation Strategy, Scale, and Feature Space on Unsupervised Methods*](https://doi.org/10.3390/rs8090761). *Remote Sensing*, 8(9), 761. | Documents sensitivity of object-level change to segmentation choices. The project does not reproduce its unsupervised method. |
| P14 | Jung, S., Lee, W. H., & Han, Y. (2021). [*Change Detection of Building Objects in High-Resolution Single-Sensor and Multi-Sensor Imagery Considering the Sun and Sensor's Elevation and Azimuth Angles*](https://doi.org/10.3390/rs13183660). *Remote Sensing*, 13(18), 3660. | Cautions that different viewing and illumination geometry can imitate building change. The project's labels remain candidate map changes. |
| P15 | Su, S., Nawata, T., & Fuse, T. (2020). [*Building Change Detection from Bitemporal Aerial Images Using Deep Learning*](https://doi.org/10.5194/isprs-annals-V-2-2020-565-2020). *ISPRS Annals*, V-2-2020, 565–571. | Context for new/demolished building outcomes. The learned detector and its accuracy are not reproduced. |
| P16 | Pekel, J.-F., Cottam, A., Gorelick, N., & Belward, A. S. (2016). [*High-Resolution Mapping of Global Surface Water and Its Long-Term Changes*](https://doi.org/10.1038/nature20584). *Nature*, 540, 418–422. | Shows why temporal water interpretation benefits from many observations. A two-date `water_lost` edge does not establish permanence or seasonality. |
| P17 | Holme, P., & Saramäki, J. (2012). [*Temporal Networks*](https://doi.org/10.1016/j.physrep.2012.03.001). *Physics Reports*, 519(3), 97–125. | General temporal-network context for timestamped nodes and directed forward-time relationships. |

## Other papers consulted for comparison or future work

These sources inform limitations, alternatives, or proposed validation. Their
models and benchmark results are **not** implemented or claimed by SatelliteGraph.

| ID | Thesis-ready reference and primary link | Why it matters / current status |
|---|---|---|
| P18 | Zhou, W., Troy, A., & Grove, M. (2008). [*Object-Based Land Cover Classification and Change Analysis in the Baltimore Metropolitan Area Using Multitemporal High Resolution Remote Sensing Data*](https://doi.org/10.3390/s8031613). *Sensors*, 8(3), 1613–1636. | Historical object-based high-resolution land-cover change precedent. |
| P19 | Gao, S., Li, W., Sun, K., Wei, J., Chen, Y., & Wang, X. (2022). [*Built-Up Area Change Detection Using Multi-Task Network with Object-Level Refinement*](https://doi.org/10.3390/rs14040957). *Remote Sensing*, 14(4), 957. | Example of learned object-level refinement; not used in current deterministic refinement. |
| P20 | Tsai, Y. H., Stow, D., & Weeks, J. (2011). [*Comparison of Object-Based Image Analysis Approaches to Mapping New Buildings in Accra, Ghana Using Multi-Temporal QuickBird Satellite Imagery*](https://doi.org/10.3390/rs3122707). *Remote Sensing*, 3(12), 2707–2726. | New-building detection under multitemporal high-resolution imagery; candidate methodological comparison. |
| P21 | Vakalopoulou, M., Karantzalos, K., Komodakis, N., & Paragios, N. (2015). [*Simultaneous Registration and Change Detection in Multitemporal, Very High Resolution Remote Sensing Data*](https://openaccess.thecvf.com/content_cvpr_workshops_2015/W13/html/Vakalopoulou_Simultaneous_Registration_and_2015_CVPR_paper.html). CVPR Workshops. | Alternative joint registration/change approach; not implemented. |
| P22 | Scheffler, D., Hollstein, A., Diedrich, H., Segl, K., & Hostert, P. (2017). [*AROSICS: An Automated and Robust Open-Source Image Co-Registration Software for Multi-Sensor Satellite Data*](https://doi.org/10.3390/rs9070676). *Remote Sensing*, 9(7), 676. | Recommended comparison for spatially varying registration error; current built-in fit is global affine. |
| P23 | Li, J., Hu, Q., & Ai, M. (2020). [*RIFT: Multi-Modal Image Matching Based on Radiation-Variation Insensitive Feature Transform*](https://doi.org/10.1109/TIP.2019.2959244). *IEEE Transactions on Image Processing*, 29, 3296–3310. | Candidate cross-sensor feature matcher to benchmark against SIFT; not run here. |
| P24 | Sarlin, P.-E., DeTone, D., Malisiewicz, T., & Rabinovich, A. (2020). [*SuperGlue: Learning Feature Matching with Graph Neural Networks*](https://openaccess.thecvf.com/content_CVPR_2020/html/Sarlin_SuperGlue_Learning_Feature_Matching_With_Graph_Neural_Networks_CVPR_2020_paper.html). CVPR, 4938–4947. | Context-aware local feature matching and unmatched decisions; not object-graph matching in this project. |
| P25 | Sun, J., Shen, Z., Wang, Y., Bao, H., & Zhou, X. (2021). [*LoFTR: Detector-Free Local Feature Matching with Transformers*](https://openaccess.thecvf.com/content/CVPR2021/html/Sun_LoFTR_Detector-Free_Local_Feature_Matching_With_Transformers_CVPR_2021_paper.html). CVPR, 8922–8931. | Candidate control-point generator for low-texture image pairs; not run here. |
| P26 | Lindenberger, P., Sarlin, P.-E., & Pollefeys, M. (2023). [*LightGlue: Local Feature Matching at Light Speed*](https://openaccess.thecvf.com/content/ICCV2023/html/Lindenberger_LightGlue_Local_Feature_Matching_at_Light_Speed_ICCV_2023_paper.html). ICCV, 17627–17638. | Candidate learned matcher benchmark; no LightGlue weights are loaded. |
| P27 | Vayer, T., Courty, N., Tavenard, R., Chapel, L., & Flamary, R. (2019). [*Optimal Transport for Structured Data with Application on Graphs*](https://proceedings.mlr.press/v97/titouan19a.html). ICML/PMLR 97, 6275–6284. | Fused Gromov–Wasserstein is a future option for node-feature and graph-structure matching; not implemented. |
| P28 | Batra, A., Singh, S., Pang, G., Basu, S., Jawahar, C. V., & Paluri, M. (2019). [*Improved Road Connectivity by Joint Learning of Orientation and Segmentation*](https://openaccess.thecvf.com/content_CVPR_2019/html/Batra_Improved_Road_Connectivity_by_Joint_Learning_of_Orientation_and_Segmentation_CVPR_2019_paper.html). CVPR, 10385–10393. | Demonstrates why road topology and orientation require dedicated evidence; no orientation head/refinement network is used here. |
| P29 | Mattyus, G., Luo, W., & Urtasun, R. (2017). [*DeepRoadMapper: Extracting Road Topology from Aerial Images*](https://openaccess.thecvf.com/content_iccv_2017/html/Mattyus_DeepRoadMapper_Extracting_Road_ICCV_2017_paper.html). ICCV, 3438–3446. | Road-centerline and missing-connection reasoning; this project currently keeps observed region adjacency only. |
| P30 | Zorzi, S., Bazrafkan, S., Habenschuss, S., & Fraundorfer, F. (2022). [*PolyWorld: Polygonal Building Extraction with Graph Neural Networks in Satellite Images*](https://openaccess.thecvf.com/content/CVPR2022/html/Zorzi_PolyWorld_Polygonal_Building_Extraction_With_Graph_Neural_Networks_in_Satellite_CVPR_2022_paper.html). CVPR, 1848–1857. | Future building-boundary comparison; its polygon predictor is not used. |
| P31 | Carvalho, O. L. F. de, Carvalho Júnior, O. A. de, Albuquerque, A. O. de, & Silva, D. G. e. (2026). [*Remote SAMsing: From Segment Anything to Segment Everything*](https://arxiv.org/abs/2605.00256). arXiv:2605.00256, preprint. | Future multi-pass, cross-tile SAM coverage comparison; not part of current tile-local Stage 2R. |

## Primary software documentation and standards

These should be cited in the methods or software statement where relevant;
they are not peer-reviewed performance evidence.

| ID | Authoritative source | Project use |
|---|---|---|
| S01 | [GDAL Geotransform Tutorial](https://gdal.org/en/stable/tutorials/geotransforms_tut.html) | Pixel-to-world affine coordinate convention and tiling metadata. |
| S02 | [Rasterio reprojection documentation](https://rasterio.readthedocs.io/en/stable/topics/reproject.html) | Resolution conversion, CRS reprojection, and yearly image/geometry alignment. |
| S03 | [GDAL raster sieve](https://gdal.org/en/stable/programs/gdal_sieve.html) | Background analogy for connected small-region replacement; the project's unknown cleanup uses its own gated implementation. |
| S04 | [scikit-image region-adjacency-graph merging example](https://scikit-image.org/docs/stable/auto_examples/segmentation/plot_rag_merge.html) | Comparison for appearance-based neighboring-region merging. |
| S05 | [OGC Simple Feature Access, Part 1](https://www.ogc.org/standards/sfa/) | Polygon/multipolygon geometry vocabulary and geospatial interchange. |
| S06 | [SciPy `linear_sum_assignment`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.linear_sum_assignment.html) | One-to-one candidate assignment with unmatched options added by project code. |
| S07 | [OpenCV geometric transformations and RANSAC API](https://docs.opencv.org/4.x/d9/d0c/group__calib3d.html) and [phase correlation API](https://docs.opencv.org/4.x/d7/df3/group__imgproc__motion.html) | Registration implementation; cite with P09/P10 for algorithm lineage. |
| S08 | [SAHI sliced-inference guide](https://github.com/obss/sahi/blob/main/docs/guides/sliced-inference.md) | Comparison for overlap postprocessing; SAHI itself is not executed. |
| S09 | [SAM 2 source repository](https://github.com/facebookresearch/sam2) | Bundled `third_party/sam2` inference code and model usage. |
| S10 | [OpenEarthMap lightweight model repository](https://github.com/cliffbb/OEM-Lightweight) | Bundled `third_party/oem-lightweight` FasterSeg inference code. |

## Inputs, attribution, and what still needs verification

The 2013 and 2023 annual artifacts, 2017/2023 tile folders, image PNGs/TIFFs,
tile metadata, and model checkpoints are project inputs, **not research
publications**. Before submitting a thesis or article, record their original
image provider, sensor, capture dates, processing level, ground sample
distance, CRS, licensing/permission, checkpoint version and source, and any
manual ground-truth labels. These fields cannot be safely inferred from a
filename or this repository alone. The full project results are candidate
mapped changes until independently labeled examples establish precision and
recall. Cite the exact software/model versions used for the reported run.

Citation corrections made during this audit: the 2021 building-change paper
at DOI `10.3390/rs13183660` is **Jung, Lee, and Han**, not “He et al.”; the
2016 segmentation-strategy paper at DOI `10.3390/rs8090761` is **Ma et al.**,
not “Lv et al.”. The source links above point to publisher, conference,
author-hosted, or official project pages. This ledger should be reviewed once
more against each publisher's final citation export when the thesis reference
style is chosen.
