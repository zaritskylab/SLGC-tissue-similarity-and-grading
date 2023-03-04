![NANOBIOPSY logo](./logo.png)

# Nondestructive Spatial Lipidomics for Glioma Classification - Tissue Similarity and Grading
Davide Alessandro Martella<sup>1</sup>, Leor Ariel Rose<sup>2</sup>, Nadia Rouatbi<sup>3</sup>, Chenlei Gu<sup>1</sup>, Valeria Caprettini<sup>1</sup>, Magnus Jensen<sup>1</sup>, Cathleen Hagemann<sup>1,4</sup>, Andrea Serio<sup>1,4</sup>, Khuloud Al-Jamal<sup>3</sup>, Mads Bergholt<sup>1</sup>, Paul Brennan<sup>5</sup>, Assaf Zaritsky<sup>2</sup>, Ciro Chiappini<sup>1,6</sup>

1. Centre for Craniofacial and Regenerative Biology, King’s College London, SE1 9RT, London, UK.
2. Department of Software and Information Systems Engineering, Ben-Gurion University of the Negev, Be'er Sheva, 8410501 Israel.
3. Institute of Pharmaceutical Sciences, King’s College London, SE1 9NQ, London, UK.
4. The Francis Crick Institute, NW1 1AT, London, UK.
5. Centre for Clinical Brain Sciences, University of Edinburgh, EH16 4SB, Edinburgh, UK.
6. London Centre for Nanotechnology, King’s College London, WC2R 2LS, London, UK.


## Repository contents
This repository contains code for the tissue similarity and grading of DESI human glioma nanobiopsy part of the paper "Nondestructive Spatial Lipidomics for Glioma Classification".

[Main Python file](./main.py) - contains the main code to run the processing, correlation and classification analysis.

[Processing folder](./processing) - contains the processing code.

[Correlation analysis folder](./correlation) - contains the correlation analysis code.

[Classification analysis folder](./classification) - contains the classification analysis code.

[Figure creation file](./visualization.ipynb) - contains the figure creation code.

## Prerequisite
This code requires the DESI-MS Human Glioma dataset (details in the paper). You should acquire the data and create a folder containing the raw MSI files and the meta data CSV file. Folder structure should be as follows:

      .
      ├── DHG
      │   ├── raw
      │   │   ├── HG 1-r.ibd
      │   │   ├── HG 1-r.ibh
      │   │   ├── HG 1-r.imzML
      │   │   ├── HG 1-s.ibd
      │   │   ├── HG 1-s.imzML
      │   │   ├── ...
      │   └── metadata.csv
      └── SLGC-tissue-similarity-and-grading

## Project setup and run:

1. Clone this repository.
2. Open cmd/shell/terminal and go to project folder: `cd SLGC-tissue-similarity-and-grading`
3. Create a conda environment: `conda env create -f environment.yml`
4. Activate the conda environment `conda activate tfgpu_jup`
5. Run `python main.py`
6. Run `visualization.ipynb` Jupyter notebook.

## Citation
If you use this code, please cite: 

```
Paper MLA
```

Please contact leor.rose@gmail.com or assafzar@gmail.com for bugs or questions.
