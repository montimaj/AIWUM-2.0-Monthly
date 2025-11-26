# Aquaculture and Irrigation Water Use Model-2.0-Monthly Software

Maintainer: [Sayantan Majumdar](https://www.dri.edu/directory/sayantan-majumdar/) [sayantan.majumdar@dri.edu]

<img src="Readme_Figures/dri-logo.png" height="45"/> &nbsp; <img src="Readme_Figures/csu-logo.png" height="55"/>

## Citations

Majumdar, S., Smith, R. G., and Hasan, M. F, 2025, A High-Resolution Data-Driven Monthly Aquaculture and Irrigation Water Use Model in the Mississippi Alluvial Plain, IGARSS 2025 - 2025 IEEE International Geoscience and Remote Sensing Symposium, 2686–2691, https://doi.org/10.1109/IGARSS55030.2025.11243173

Majumdar, S., Smith, R. G., Hasan, M. F., Wilson, J. L., White, V. E., Bristow, E. L., Rigby, J. R., Kress, W. H., and Painter, J. A., 2024, Improving crop-specific groundwater use estimation in the Mississippi Alluvial Plain: Implications for integrated remote sensing and machine learning approaches in data-scarce regions, Journal of Hydrology: Regional Studies, 52, 101674, https://doi.org/10.1016/j.ejrh.2024.101674

Majumdar, S., Smith, R.G., Hasan, M.F., Wilson, J.L., White, V.E., Bristow, E.L., Rigby, J.R., Kress, W.H., and Painter, J.A., 2024, Aquaculture and Irrigation Water Use Model 2.0 software, U.S. Geological Survey software release, https://doi.org/10.5066/P137FIUZ.

## Summary

The Mississippi Alluvial Plain (MAP) is one of the most productive agricultural regions in the US and extracts more than 11 km<sup>3</sup>/year for irrigation activities. The heavy drivers of groundwater use are aquaculture and crops, which include rice, cotton, corn, and soybeans ([Wilson, 2021](https://doi.org/10.3133/sir20215011)). Consequently, groundwater-level declines in the MAP region ([Clark and others, 2011](https://doi.org/10.3133/pp1785)) pose a substantial challenge to water sustainability, and hence, we need reliable groundwater pumping monitoring solutions to manage this resource appropriately.

Here, we first summarize the Aquaculture and Irrigation Water Use Model (AIWUM) 2.0 ([Majumdar and others, 2024a](https://doi.org/10.1016/j.ejrh.2024.101674)) and then describe the changes made in AIWUM 2.0 Monthly.

### AIWUM 2.0

In AIWUM 2.0, we used remote sensing datasets and machine learning to improve AIWUM 1.1, previously developed by the U.S. Geological Survey (USGS) ([Wilson, 2021](https://doi.org/10.3133/sir20215011); [Bristow and Wilson, 2023](https://doi.org/10.5066/P9RGZOBZ)). Unlike AIWUM 1.1, which assigned annual irrigation water use per acre based on average water use for each crop type, AIWUM 2.0 integrated remote sensing data in a machine learning framework to produce improved estimates of annual and monthly groundwater use at 1-km spatial resolution.  

By leveraging machine learning, we were able to automatically relate water balance components (e.g., precipitation, evapotranspiration, soil moisture, and others) and in-situ groundwater use data, which was not possible before in the MAP. This was the first study to develop a groundwater use estimation model at the field scale by linking the point of use (flowmeter) to the application area (individual fields), which was then used to produce spatially continuous (1-km) groundwater use predictions from 2014-2020. Ultimately, this machine learning-based water-use model is one part of an ongoing USGS effort to build a hydrologic decision-support system specific to the MAP region. Thus, AIWUM 2.0 addressed a critical gap in the hydrologic modeling of this region by introducing a robust machine learning-based method for groundwater use estimation.

Our goal was to include datasets that are correlated to groundwater use, such as remotely sensed evapotranspiration, model-based precipitation estimates, and air temperature. Machine learning provides a more robust method to predict groundwater use than traditional process-based models because these algorithms can handle the complex and often non-linear relationships between explanatory and response variables ([Hastie and others, 2001](https://hastie.su.domains/ElemStatLearn/printings/ESLII_print12_toc.pdf)). Although the overall approach to predicting groundwater use was similar to our efforts in Kansas ([Majumdar and others, 2020](https://doi.org/10.1029/2020WR028059)) and Arizona ([Majumdar and others, 2022](https://doi.org/10.1002/hyp.14757)), modeling groundwater use in the MAP region is more complex because of limited metered data availability. Hence, we incorporated additional steps to pre-process the datasets, e.g., linking the flowmeters to the fields using K-Dimensional trees ([Rosenberg, 1985](https://doi.org/10.1109/TCAD.1985.1270098)). We also used more predictors compared to the Kansas and Arizona studies because of the lack of training data and to better explain the scatter in the model predictions.

For AIWUM 2.0, we employed Distributed Random Forests (DRF) ([Breiman, 2001](https://doi.org/10.1023/A:1010933404324); [Ke and others, 2017](https://dl.acm.org/doi/10.5555/3294996.3295074)), a distributed ensemble machine learning algorithm to predict annual groundwater use (2014-2020) across the Mississippi embayment (MISE), a large area overlapping the MAP region, using the following datasets as predictors: field centroids (latitude, longitude), SSEBop (Operational Simplified Surface Energy Balance) evapotranspiration ([Senay and others, 2013](https://doi.org/10.1111/jawr.12057)), precipitation and air temperature from PRISM (Parameter-elevation Regressions on Elevation Slopes Model) ([Daly and others, 2008](http://doi.wiley.com/10.1002/joc.1688)), crop type from the USDA-NASS CDL (US Department of Agriculture- National Agricultural Statistics Service Cropland Data Layer) ([Boryan and others, 2011](https://doi.org/10.1080/10106049.2011.562309)), surface runoff and soil moisture change from TerraClimate ([Abatzoglou and others, 2018](https://doi.org/10.1038/sdata.2017.191)), and Soil-Water Balance (SWB) hydrologic soil group (HSG) and irrigation ([Westenbroek and others, 2021](https://doi.org/10.3133/ofr20211008)). 

We relied on the annual groundwater use data provided by the Mississippi Department of Environmental Quality (MDEQ) to train and validate our model. The MDEQ collects these crop-specific data through the Delta Voluntary Metering Program (DVMP) and performs quality assurance and quality control (QA/QC) checks ([Wilson, 2021](https://doi.org/10.3133/sir20215011)). Crop-specific weights derived from the USGS real-time flowmeter data ([Majumdar and others, 2024b](https://doi.org/10.5066/P9CET25K)) were used to disaggregate these annual-scale predictions into monthly timesteps. 

A table summarizing the predictor variables and a diagram of the general processing workflow of AIWUM 2.0 can be found in [Majumdar and others (2024c)](https://doi.org/10.5066/P137FIUZ). For details on the MAP extent and its generalized regions (the Mississippi Delta being one of them), see [Painter and Westerman (2018)](https://doi.org/10.5066/F70R9NMJ) and [Ladd and Travers (2019)](https://doi.org/10.5066/P915ZZQM), respectively.  

### Changes in AIWUM 2.0 Monthly

Unlike AIWUM 2.0 which predicts both annual and monthly groundwater use, AIWUM 2.0 Monthly predicts monthly groundwater use at the same 1-km spatial resolution. Here, we aim to incorporate the USGS real-time flowmeter data available across the MAP to disaggegate the annual VMP data into monthly timesteps and increase our training samples. The disaggregation process is described below.

#### Disaggregating VMP data with the real-time weights
For the annual model, we disaggregated with AIWUM 2.0 predictions ([Majumdar and others, 2024b](https://doi.org/10.5066/P9CET25K)) with the normalized crop-specific weights derived from the real-time flowmeter data. Here, we apply the same approach as used in [Majumdar and others (2024a)](https://doi.org/10.1016/j.ejrh.2024.101674) to 
disaggregate the original VMP data. Therefore, we develop monthly VMP data from the annual one and then include the aggregated monthly real-time measurements as well. One caveat of this approach is that 
the disaggregated VMP data do not represent the actual ground truth and can be conceptualized as 'simulated data'.

We use the real-time data (2018-2021) and calculate the normalized weights for each month. We then multiply these weights to the annual VMP data to get the disaggregated annual groundwater use values.

The predictor variables include latitude, longitude, crop type, precipitation, maximum temperature, total evapotranspiration estimated with SSEBop, surface run-off and soil moisture (TerraClimate). Note that, although we do not use the SWB datasets (HSG-derived infiltration rate and irrigation demands) in AIWUM 2.0 Monthly, we have the provision to include these as model predictors in the software. A table summary of the predictor variables for AIWUM 2.0 Monthly is given below.

| Variable           | Operator       | Time Period Sampled | Spatial Resolution (m) | Source                                                                   | Additional Processing Notes                                                                              |
|--------------------|----------------|---------------------|------------------------|--------------------------------------------------------------------------|----------------------------------------------------------------------------------------------------------|
| Latitude           | ---            | 2014-2021           | ---                    | [Majumdar and others (2024)](https://doi.org/10.5066/P9CET25K)           | Used yearly permitted boundaries and K-Dimensional Tree to get field centroids                           |
| Longitude          | ---            | 2014-2021           | ---                    | [Majumdar and others (2024)](https://doi.org/10.5066/P9CET25K)           | Used yearly permitted boundaries and K-Dimensional Tree to get field centroids                           |
| Crop type          | Mode (spatial) | Entire year         | 30                     | [Boryan and others (2011)](https://doi.org/10.1080/10106049.2011.562309) | Data were up-sampled to 0.1 km grid as in [Bristow and Wilson (2023)](https://doi.org/10.5066/P9RGZOBZ). |
| Precipitation      | ---            | Entire year         | 800                    | [Daly and others (2008)](http://doi.wiley.com/10.1002/joc.1688)          | This dataset is available at the monthly scale.                                                          |
| Temperature        | ---            | Entire year         | 800                    | [Daly and others (2008)](http://doi.wiley.com/10.1002/joc.1688)          | This dataset is available at the monthly scale.                                                          |
| Evapotranspiration | ---            | Entire year         | 1000                   | [Senay and others (2013)](https://doi.org/10.1111/jawr.12057)            | This dataset is available at the monthly scale.                                                          |
| Surface run-off    | ---            | Entire year         | ~ 4,000                | [Abatzoglou and others (2018)](https://doi.org/10.1038/sdata.2017.191)   | This dataset is available at the monthly scale.                                                          |
| Soil Moisture      | ---            | Entire year         | ~ 4000                 | [Abatzoglou and others (2018)](https://doi.org/10.1038/sdata.2017.191)   | This dataset is available at the monthly scale.                                                          |

See the [IGARSS 2025 poster](poster/IGARSS2025_Poster_SM.pdf) and [Majumdar and others, 2025](https://doi.org/10.1109/IGARSS55030.2025.11242748) for further details.

## Getting Started

[Installing the correct environment and running the project](aiwum2_monthly/README.md)

## Related External Resources
Abatzoglou, J.T., Dobrowski, S.Z., Parks, S.A., and Hegewisch, K.C., 2018, TerraClimate, a high-resolution global dataset of monthly climate and climatic water balance from 1958–2015, Scientific Data, 5(1), 170191, https://doi.org/10.1038/sdata.2017.191.

Anaconda Software Distribution, 2023, Anaconda Documentation, Anaconda Inc., https://docs.anaconda.com/.

Asfaw, D., Smith, R. G., Majumdar, S., Grote, K., Fang, B., Wilson, B. B., Lakshmi, V., and Butler, J. J., 2025, Predicting groundwater withdrawals using machine learning with limited metering data: Assessment of training data requirements, Agricultural Water Management, 318, 109691, https://doi.org/10.1016/j.agwat.2025.109691

Boryan, C., Yang, Z., Mueller, R., and Craig, M., 2011, Monitoring US agriculture: the US Department of Agriculture, National Agricultural Statistics Service, Cropland Data Layer Program, Geocarto International, 26(5), 341–358, https://doi.org/10.1080/10106049.2011.562309.

Breiman, L., 2001, Random Forests, Machine Learning, 45(1), 5–32, https://doi.org/10.1023/A:1010933404324.

Bristow, E.L., and Wilson, J.L., 2023, Aquaculture and irrigation water-Use model (AIWUM) version 1.1 estimates and related datasets for the Mississippi Alluvial Plain: U.S. Geological Survey data release, https://doi.org/10.5066/P9RGZOBZ.

Clark, B.R., Hart, R.M., and Gurdak, J.J., 2011, Groundwater availability of the Mississippi embayment: U.S. Geological Survey Professional Paper 1785, 62 p., https://doi.org/10.3133/pp1785.

Daly, C., Halbleib, M., Smith, J.I., Gibson, W.P., Doggett, M.K., Taylor, G.H., Curtis, J., and Pasteris, P.P., 2008, Physiographically sensitive mapping of climatological temperature and precipitation across the conterminous United States, International Journal of Climatology, 28(15), 2031–2064, https://doi.org/10.1002/joc.1688.

Hasan, M. F., Smith, R. G., Majumdar, S., Huntington, J. L., Alves Meira Neto, A., and Minor, B. A., 2025, Satellite data and physics-constrained machine learning for estimating effective precipitation in the Western United States and application for monitoring groundwater irrigation, Agricultural Water Management, 319, 109821, https://doi.org/10.1016/j.agwat.2025.109821

Hastie, T., Tibshirani, R., & Friedman, J., 2001, The Elements of Statistical Learning: Data Mining, Inference, and Prediction, Springer New York.

Ke, G., Meng, Q., Finley, T., Wang, T., Chen, W., Ma, W., Ye, Q., and Liu, T.-Y., 2017, LightGBM: A Highly Efficient Gradient Boosting Decision Tree. In I. Guyon, U. V Luxburg, S. Bengio, H. Wallach, R. Fergus, S. Vishwanathan, & R. Garnett (Eds.), Advances in Neural Information Processing Systems (Vol. 30), Curran Associates, Inc., https://dl.acm.org/doi/10.5555/3294996.3295074.

Ladd, D.E., and Travers, L.R., 2019, Generalized regions of the Mississippi Alluvial Plain, U.S. Geological Survey data release, https://doi.org/10.5066/P915ZZQM.

Majumdar, S., Smith, R., Butler, J.J., and Lakshmi, V., 2020, Groundwater withdrawal prediction using integrated multitemporal remote sensing data sets and machine learning, Water Resources Research, 56(11), e2020WR028059, https://doi.org/10.1029/2020WR028059.

Majumdar, S., Smith, R., Conway, B.D., and Lakshmi, V., 2022, Advancing remote sensing and machine learning‐driven frameworks for groundwater withdrawal estimation in Arizona: Linking land subsidence to groundwater withdrawals, Hydrological Processes, 36(11), e14757, https://doi.org/10.1002/hyp.14757.

Majumdar, S., Smith, R.G., Hasan, M.F., Wilson, J.L., White, V.E., Bristow, E.L., Rigby, J.R., Kress, W.H., and Painter, J.A., 2024a, Improving crop-specific groundwater use estimation in the Mississippi Alluvial Plain: Implications for integrated remote sensing and machine learning approaches in data-scarce regions, Journal of Hydrology: Regional Studies, 52, 101674, https://doi.org/10.1016/j.ejrh.2024.101674.

Majumdar, S., Smith, R.G., Hasan, M.F., Wilson, J.L., White, V.E., Bristow, E.L., Rigby, J.R., Kress, W.H., and Painter, J.A., 2024b, Aquaculture and Irrigation Water Use Model (AIWUM) 2.0 input and output datasets, U.S. Geological Survey data release, https://doi.org/10.5066/P9CET25K.

Majumdar, S., Smith, R.G., Hasan, M.F., Wilson, J.L., White, V.E., Bristow, E.L., Rigby, J.R., Kress, W.H., and Painter, J.A., 2024c, Aquaculture and Irrigation Water Use Model 2.0 software, U.S. Geological Survey software release, https://doi.org/10.5066/P137FIUZ.

Ott, T. J., Majumdar, S., Huntington, J. L., Pearson, C., Bromley, M., Minor, B. A., ReVelle, P., Morton, C. G., Sueki, S., Beamer, J. P., and Jasoni, R. L, 2024, Toward field-scale groundwater pumping and improved groundwater management using remote sensing and climate data, Agricultural Water Management, 302, 109000. https://doi.org/10.1016/j.agwat.2024.109000

Painter, J.A., and Westerman, D.A., 2018, Mississippi Alluvial Plain extent, November 2017: U.S. Geological Survey data release, https://doi.org/10.5066/F70R9NMJ.

Rosenberg, J.B., 1985, Geographical Data Structures Compared: A Study of Data Structures Supporting Region Queries, IEEE Transactions on Computer-Aided Design of Integrated Circuits and Systems, 4(1), 53–67, https://doi.org/10.1109/TCAD.1985.1270098.

Senay, G.B., Bohms, S., Singh, R.K., Gowda, P.H., Velpuri, N.M., Alemu, H., and Verdin, J.P., 2013, Operational Evapotranspiration Mapping Using Remote Sensing and Weather Datasets: A New Parameterization for the SSEB Approach, JAWRA Journal of the American Water Resources Association, 49(3), 577–591, https://doi.org/10.1111/jawr.12057.

Wilson, J.L., 2021, Aquaculture and Irrigation Water-Use Model (AIWUM) version 1.0—An agricultural water-use model developed for the Mississippi Alluvial Plain, 1999–2017: U.S. Geological Survey Scientific Investigations Report 2021–5011, 36 p., https://doi.org/10.3133/sir20215011.