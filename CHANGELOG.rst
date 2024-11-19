Changelog
=========

[1.0.0] (2024-11-XX)
--------------------
Added
*******
- support for L2A product level
- example notebooks
- accuracy assessment and comparison with previous models

Changed
*******
- retrained all models with new architecture and training data
- removed backwards compatibility with old band naming scheme

[0.2.2] (2024-10-21)
--------------------
Added
*******
- expose onnxruntime execution provider

[0.2.1] (2023-12-05)
--------------------
Added
*******
- support onnxruntime session options

Changed
*******
- band naming scheme to match STAC standard

Fixed
*******
- version mismatch with pip release

[0.2.0] (2023-05-24)
--------------------
Added
*******
- 4 band model to manifest

[0.1.9] (2023-05-12)
--------------------
Added
*******
- support for 4 band (R-G-B-NIR) images

[0.1.8] (2023-01-30)
--------------------
Added
*******
- warn user when smaller tiling than 256x256 is used

[0.1.7] (2022-12-09)
--------------------
Added
*******
- added support for onnxruntime-openvino provider

[0.1.6] (2022-11-30)
--------------------
Changed
*******
- update onnxruntime version and add providers to session

[0.1.5] (2022-11-11)
--------------------
Changed
*******
- enables setting the buffer size of invalid areas via `invalid_buffer` parameter

[0.1.4]  (2022-03-03)
----------------------
Changed
*******
- removed `pytest` from normal requirements

[0.1.3]  (2021-10-11)
----------------------
Added
*****
- added DOI

Changed
*******
- removed `beta`

[0.1.2-beta]  (2021-01-21)
---------------------

Fixed
*******
- memory error when running inference on GPU

Changed
*******
- updated README


[0.1.1-beta]  (2021-01-20)
---------------------

Fixed
*******
- missing datafile for onnx model


[0.1.0-beta]  (2021-01-20)
---------------------

Fixed
*******
- relative model import
- corrections in README and pip descriptions


[0.0.1-alpha]  (2021-01-19)
---------------------

- first release
