import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

def _radiology_paper_ground_truth():
    return {
        "radimagenet": {
            "title": "RadImageNet: An Open Radiologic Deep Learning Research Dataset for Effective Transfer Learning",
            "abstract": (
                "Purpose: To demonstrate the value of pretraining with millions of radiologic images compared "
                "with ImageNet photographic images on downstream medical applications when using transfer "
                "learning. Materials and methods: This retrospective study included patients who underwent a "
                "radiologic study between 2005 and 2020 at an outpatient imaging facility. Key images and "
                "associated labels from the studies were retrospectively extracted from the original study "
                "interpretation. These images were used for RadImageNet model training with random weight "
                "initiation. The RadImageNet models were compared with ImageNet models using the area under the "
                "receiver operating characteristic curve (AUC) for eight classification tasks and using Dice "
                "scores for two segmentation problems. Results: The RadImageNet database consists of 1.35 "
                "million annotated medical images in 131 872 patients who underwent CT, MRI, and US for "
                "musculoskeletal, neurologic, oncologic, gastrointestinal, endocrine, abdominal, and pulmonary "
                "pathologic conditions."
            ),
            "link": "https://doi.org/10.1148/ryai.210315",
            "date": "27-July-2022",
            "pmid": "36204533",
            "expected": {
                "name": "RadImageNet",
                "num_images": 1350000,
                "num_patients": 131872,
                "modalities": ["CT", "MRI", "US"],
                "body_regions": ["brain", "chest", "abdomen", "pelvis", "limbs"],
                "additional_data": ["segmentation"],
            },
            "name_aliases": ["radimagenet"],
        },
        "mimic_cxr": {
            "title": "MIMIC-CXR, a de-identified publicly available database of chest radiographs with free-text reports",
            "abstract": (
                "Chest radiography is an extremely powerful imaging modality, allowing for a detailed inspection "
                "of a patient's chest, but requires specialized training for proper interpretation. With the advent "
                "of high performance general purpose computer vision algorithms, the accurate automated analysis of "
                "chest radiographs is becoming increasingly of interest to researchers. Here we describe MIMIC-CXR, "
                "a large dataset of 227,835 imaging studies for 65,379 patients presenting to the Beth Israel Deaconess "
                "Medical Center Emergency Department between 2011-2016. Each imaging study can contain one or more images, "
                "usually a frontal view and a lateral view. A total of 377,110 images are available in the dataset. "
                "Studies are made available with a semi-structured free-text radiology report that describes the "
                "radiological findings of the images, written by a practicing radiologist contemporaneously during routine "
                "clinical care. All images and reports have been de-identified to protect patient privacy. The dataset is "
                "made freely available to facilitate and encourage a wide range of research in computer vision, natural "
                "language processing, and clinical data mining."
            ),
            "link": "https://doi.org/10.1038/s41597-019-0322-0",
            "date": "12-December-2019",
            "pmid": "31831740",
            "expected": {
                "name": "MIMIC-CXR",
                "num_images": 377110,
                "num_patients": 65379,
                "modalities": ["X-ray"],
                "body_regions": ["chest"],
                "additional_data": ["reports"],
            },
            "name_aliases": ["mimic", "mimiccxr", "mimic-cxr"],
        },
        "uk_biobank": {
            "title": "The UK Biobank imaging enhancement of 100,000 participants: rationale, data collection, management and future directions",
            "abstract": (
                "UK Biobank is a population-based cohort of half a million participants aged 40-69 years recruited "
                "between 2006 and 2010. In 2014, UK Biobank started the world's largest multi-modal imaging study, with "
                "the aim of re-inviting 100,000 participants to undergo brain, cardiac and abdominal magnetic resonance "
                "imaging, dual-energy X-ray absorptiometry and carotid ultrasound. The combination of large-scale "
                "multi-modal imaging with extensive phenotypic and genetic data offers an unprecedented resource for "
                "scientists to conduct health-related research. This article provides an in-depth overview of the imaging "
                "enhancement, including the data collected, how it is managed and processed, and future directions."
            ),
            "link": "https://doi.org/10.1038/s41467-020-15948-9",
            "date": "26-May-2020",
            "pmid": "32457287",
            "expected": {
                "name": "UK Biobank",
                "num_images": None,
                "num_patients": 100000,
                "modalities": ["MRI"],
                "body_regions": ["brain"],
                "additional_data": [],
            },
            "name_aliases": ["ukbiobank", "uk biobank"],
        },
        "tcia": {
            "title": "The Cancer Imaging Archive (TCIA): Maintaining and Operating a Public Information Repository",
            "abstract": (
                "The National Institutes of Health have placed significant emphasis on sharing of research data to "
                "support secondary research. Investigators have been encouraged to publish their clinical and imaging "
                "data as part of fulfilling their grant obligations. Realizing it was not sufficient to merely ask "
                "investigators to publish their collection of imaging and clinical data, the National Cancer Institute "
                "(NCI) created the open source National Biomedical Image Archive software package as a mechanism for "
                "centralized hosting of cancer related imaging. NCI has contracted with Washington University in Saint "
                "Louis to create The Cancer Imaging Archive (TCIA)-an open-source, open-access information resource to "
                "support research, development, and educational initiatives utilizing advanced medical imaging of cancer. "
                "In its first year of operation, TCIA accumulated 23 collections (3.3 million images). Operating and "
                "maintaining a high-availability image archive is a complex challenge involving varied archive-specific "
                "resources and driven by the needs of both image submitters and image consumers. Quality archives of any "
                "type (traditional library, PubMed, refereed journals) require management and customer service. This paper "
                "describes the management tasks and user support model for TCIA."
            ),
            "link": "https://doi.org/10.1007/s10278-013-9622-7",
            "date": "01-December-2013",  # "25-July-2013",
            "pmid": "23884657",
            "expected": {
                "name": "The Cancer Imaging Archive (TCIA)",
                "num_images": 3300000,
                "num_patients": None,
                "modalities": ["CT", "MRI", "X-ray", "US", "PET"],
                "body_regions": ["brain", "chest", "abdomen", "pelvis", "limbs"],
                "additional_data": ["reports", "captions", "segmentation"],
            },
            "name_aliases": ["tcia", "cancer imaging archive"],
        },
        "merlin": {
            "title": "Merlin: a computed tomography vision-language foundation model and dataset",
            "abstract": (
                "The large volume of abdominal computed tomography (CT) scans coupled with the shortage of radiologists have intensified the need for automated medical image analysis tools. Previous state-of-the-art approaches for automated analysis leverage vision–language models (VLMs) that jointly model images and radiology reports. However, current medical VLMs are generally limited to 2D images and short reports. Here to overcome these shortcomings for abdominal CT interpretation, we introduce Merlin, a 3D VLM that learns from volumetric CT scans, electronic health record data and radiology reports. This approach is enabled by a multistage pretraining framework that does not require additional manual annotations. We trained Merlin using a high-quality clinical dataset of paired CT scans (>6 million images from 15,331 CT scans), diagnosis codes (>1.8 million codes) and radiology reports (>6 million tokens). We comprehensively evaluated Merlin on 6 task types and 752 individual tasks that covered diagnostic, prognostic and quality-related tasks. The non-adapted (off-the-shelf) tasks included zero-shot classification of findings (30 findings), phenotype classification (692 phenotypes) and zero-shot cross-modal retrieval (image-to-findings and image-to-impression). The model-adapted tasks included 5-year chronic disease prediction (6 diseases), radiology report generation and 3D semantic segmentation (20 organs). We validated Merlin at scale, with internal testing on 5,137 CT scans and external testing on 44,098 CT scans from 3 independent sites and 2 public datasets. The results demonstrated high generalization across institutions and anatomies. Merlin outperformed 2D VLMs, CT foundation models and off-the-shelf radiology models. We also computed scaling laws and conducted ablation studies to identify optimal training strategies. We release our trained models, code and dataset for 25,494 pairs of abdominal CT scans and radiology reports. Our results demonstrate how Merlin may assist in the interpretation of abdominal CT scans and mitigate the burden on radiologists while simultaneously adding value for future biomarker discovery and disease risk stratification."
            ),
            "link": "https://doi.org/10.1038/s41586-026-10181-8",
            "date": "04-March-2026",
            "pmid": "41781626",
            "expected": {
                "name": "Merlin",
                "num_images": None,
                "num_patients": None,
                "modalities": ["CT"],
                "body_regions": [],
                "additional_data": [],
            },
            "name_aliases": ["merlin"],
        },
        "radgenome_chest_ct": {
            "title": "Development of a large-scale grounded vision language dataset for chest CT analysis",
            "abstract": (
                "Developing generalist foundation model has recently attracted tremendous attention in the field of AI for Medicine, which requires open-source medical image datasets that incorporate diverse supervision signals across various imaging modalities. In this paper, we introduce RadGenome-Chest CT, a comprehensive, large-scale, region-guided 3D chest CT interpretation dataset based on CT-RATE. Specifically, we leverage the latest powerful universal segmentation model and large language models, to extend the original datasets from the following aspects: organ-level segmentation masks covering 197 categories, which provide intermediate reasoning visual clues for interpretation; 665K multigranularity grounded reports, where each sentence of the report is linked to the corresponding anatomical region of CT volume with a segmentation mask; 1.2M grounded VQA pairs, where questions and answers are all linked with reference segmentation masks, enabling models to associate visual evidence with textual explanations. We believe that RadGenome-Chest CT can significantly advance the development of multimodal medical foundation models, by training to generate texts based on given segmentation regions, which is unattainable with previous relevant datasets."
            ),
            "link": "https://doi.org/10.1038/s41597-025-05922-9",
            "date": "2025",
            "pmid": "41073455",
            "name_aliases": ["radgenome_chest_ct", "radgenome", "radgenomechestct", "radgenome chest ct"],
        },
        "ct-rate": {
            "title": "Generalist foundation models from a multimodal dataset for 3D computed tomography",
            "abstract": (
                "Advancements in medical imaging AI, particularly in 3D imaging, have been limited due to the scarcity of comprehensive datasets. We introduce CT-RATE, a public dataset that pairs 3D medical images with corresponding textual reports. CT-RATE comprises 25,692 non-contrast 3D chest CT scans from 21,304 unique patients. Each scan is accompanied by its corresponding radiology report. Leveraging CT-RATE, we develop CT-CLIP, a CT-focused contrastive language-image pretraining framework designed for broad applications without the need for task-specific training. We demonstrate how CT-CLIP can be used in multi-abnormality detection and case retrieval, and outperforms state-of-the-art fully supervised models across all key metrics. By combining CT-CLIP's vision encoder with a pretrained large language model, we create CT-CHAT, a vision-language foundational chat model for 3D chest CT volumes. Fine-tuned on over 2.7 million question-answer pairs derived from the CT-RATE dataset, CT-CHAT underscores the necessity for specialized methods in 3D medical imaging. Collectively, the open-source release of CT-RATE, CT-CLIP and CT-CHAT not only addresses critical challenges in 3D medical imaging but also lays the groundwork for future innovations in medical AI and improved patient care."
            ),
            "link": "https://doi.org/10.1038/s41551-025-01599-y",
            "date": "2025",
            "pmid": "41680439",
            "name_aliases": ["ct-rate", "ctrate"],
        },
        "roco": {
            "title": "ROCOv2: Radiology Objects in COntext Version 2, an Updated Multimodal Image Dataset",
            "abstract": (
                "Automated medical image analysis systems often require large amounts of training data with high quality labels, which are difficult and time consuming to generate. This paper introduces Radiology Object in COntext version 2 (ROCOv2), a multimodal dataset consisting of radiological images and associated medical concepts and captions extracted from the PMC Open Access subset. It is an updated version of the ROCO dataset published in 2018, and adds 35,705 new images added to PMC since 2018. It further provides manually curated concepts for imaging modalities with additional anatomical and directional concepts for X-rays. The dataset consists of 79,789 images and has been used, with minor modifications, in the concept detection and caption prediction tasks of ImageCLEFmedical Caption 2023. The dataset is suitable for training image annotation models based on image-caption pairs, or for multi-label image classification using Unified Medical Language System (UMLS) concepts provided with each image. In addition, it can serve for pre-training of medical domain models, and evaluation of deep learning models for multi-task learning."
            ),
            "link": "https://doi.org/10.1038/s41597-024-03496-6",
            "date": "2018",
            "pmid": "38926396",
            "name_aliases": ["roco", "roco v2", "roco2"],
        },
        "medmnist_v2": {
            "title": "MedMNIST v2: A large-scale lightweight benchmark for 2D and 3D biomedical image classification",
            "abstract": (
                "We introduce MedMNIST v2, a large-scale MNIST-like dataset collection of standardized biomedical images, including 12 datasets for 2D and 6 datasets for 3D. All images are pre-processed into a small size of 28 × 28 (2D) or 28 × 28 × 28 (3D) with the corresponding classification labels so that no background knowledge is required for users. Covering primary data modalities in biomedical images, MedMNIST v2 is designed to perform classification on lightweight 2D and 3D images with various dataset scales (from 100 to 100,000) and diverse tasks (binary/multi-class, ordinal regression, and multi-label). The resulting dataset, consisting of 708,069 2D images and 9,998 3D images in total, could support numerous research/educational purposes in biomedical image analysis, computer vision, and machine learning. We benchmark several baseline methods on MedMNIST v2, including 2D/3D neural networks and opensource/commercial AutoML tools. The data and code are publicly available at https://medmnist.com/."
            ),
            "link": "https://doi.org/10.1038/s41597-022-01721-8",
            "date": "2023",
            "pmid": "36658144",
            "name_aliases": ["medmnist", "med mnist", "medmnistv2", "med mnist v2"],
        },
        # "kits19": {
        #     "title": "The KiTS19 Challenge Data: 300 Kidney Tumor Cases with Clinical Context, CT Semantic Segmentations, and Surgical Outcomes",
        #     "abstract": (
        #         "Characterization of the relationship between a kidney tumor’s appearance on cross-sectional imaging and it’s treatment outcomes is a promising direction for informing treatement decisions and improving patient outcomes. Unfortunately, the rigorous study of tumor morphology is limited by the laborious and noisy process of making manual radiographic measurements. Semantic segmentation of the tumor and surrounding organ offers a precise quantitative description of that morphology, but it too requires significant manual effort. A large publicly available dataset of high-fidelity semantic segmentations along with clinical context and treatment outcomes could accelerate not only the study of how morphology relates to outcomes, but also the development of automatic semantic segmentation systems which could enable such studies on unprecedented scales. We present the KiTS19 challenge dataset: a collection of segmented CT imaging and treatment outcomes for 300 patients treated with partial or radical nephrectomy between 2010 and 2018. 210 of these cases have been released publicly and the remaining 90 remain private for the objective evaluation of prediction systems developed using the public cases."
        #     ),
        #     "link": "https://arxiv.org/abs/1904.00445",
        #     "date": "2020",
        #     "pmid": None,
        # },
        "idc": {
            "title": "National Cancer Institute Imaging Data Commons: Toward Transparency, Reproducibility, and Scalability in Imaging Artificial Intelligence",
            "abstract": (
                "The remarkable advances of artificial intelligence (AI) technology are revolutionizing established approaches to the acquisition, interpretation, and analysis of biomedical imaging data. Development, validation, and continuous refinement of AI tools requires easy access to large high-quality annotated datasets, which are both representative and diverse. The National Cancer Institute (NCI) Imaging Data Commons (IDC) hosts large and diverse publicly available cancer image data collections. By harmonizing all data based on industry standards and colocalizing it with analysis and exploration resources, the IDC aims to facilitate the development, validation, and clinical translation of AI tools and address the well-documented challenges of establishing reproducible and transparent AI processing pipelines. Balanced use of established commercial products with open-source solutions, interconnected by standard interfaces, provides value and performance, while preserving sufficient agility to address the evolving needs of the research community. Emphasis on the development of tools, use cases to demonstrate the utility of uniform data representation, and cloud-based analysis aim to ease adoption and help define best practices. Integration with other data in the broader NCI Cancer Research Data Commons infrastructure opens opportunities for multiomics studies incorporating imaging data to further empower the research community to accelerate breakthroughs in cancer detection, diagnosis, and treatment."
            ),
            "link": "https://doi.org/10.1148/rg.230180",
            "date": "2023",
            "pmid": "37999984",
            "name_aliases": ["idc", "idc v1", "idc v2"]
        }
    }

def _scrnaseq_paper_ground_truth():
    return {
        "tabula_sapiens": {
            "title": "Tabula Sapiens: A multiple-organ, single-cell transcriptomic atlas of humans",
            "abstract": (
                "We generated a cross-tissue single-cell transcriptomic atlas from multiple healthy human donors, "
                "profiling hundreds of thousands of cells across many organs using complementary single-cell RNA-seq "
                "protocols to support broad benchmarking and biological discovery."
            ),
            "link": "https://doi.org/10.1126/science.abl4896",
            "date": "2022",
            "pmid": "35549404",
            "expected": {
                "name": "Tabula Sapiens",
                "num_patients": 15,
                "sequencing_technology": ["10X", "SMART-Seq/SMARTSEQ"],
                "disease": ["healthy"],
                "species": ["human"],
                "tissue": ["multi-organ"],
                "cell_or_nuclei": "cell",
                "dataset_link": None,
            },
            "name_aliases": ["tabulasapiens", "tabula sapiens"],
        }
    }

def _bulk_genomics_paper_ground_truth():
    return {
        "tcga": {
            "title": "The Cancer Genome Atlas Pan-Cancer analysis project",
            "abstract": (
                "We report integrative analyses across many tumor types using multi-platform genomic profiling data, "
                "including whole-genome, exome, and bulk transcriptomic measurements, to create a broadly reusable "
                "resource for cancer genomics research."
            ),
            "link": "https://doi.org/10.1038/ng.2764",
            "date": "2013",
            "pmid": "24071849",
            "expected": {
                "name": "TCGA",
                "num_patients": 10000,
                "modalities": ["WGS", "WXS", "bulk RNA"],
                "raw_reads_available": True,
                "disease": ["cancer"],
                "species": ["human"],
                "tissue": ["multi-tissue"],
                "dataset_link": "https://portal.gdc.cancer.gov/",
            },
            "name_aliases": ["tcga", "cancer genome atlas"],
        }
    }

def _spatial_transcriptomics_paper_ground_truth():
    return {
        "stereoseq_embryo": {
            "title": "Spatiotemporal transcriptomic atlas of mouse organogenesis using DNA nanoball patterned arrays",
            "abstract": (
                "We generated a high-resolution spatial transcriptomic atlas across developmental stages with "
                "Stereo-seq, enabling systematic characterization of tissue organization and cell-state transitions "
                "during mouse organogenesis."
            ),
            "link": "https://doi.org/10.1016/j.cell.2022.04.003",
            "date": "2022",
            "pmid": "35568000",
            "expected": {
                "name": "Stereo-seq organogenesis atlas",
                "num_patients": None,
                "sequencing_technology": ["Stereo-seq"],
                "disease": [],
                "species": ["mouse"],
                "tissue": ["multi-organ"],
                "cell_or_nuclei": "cell",
                "dataset_link": None,
            },
            "name_aliases": ["stereoseq", "stereo-seq"],
        }
    }

#* add additional ground truth papers for other modalities here, e.g. genomics, pathology, etc


def _paper_irrelevant():
    return {
        "seurat_vs_scanpy": {
            "title": "The impact of package selection and versioning on single-cell RNA-seq analysis",
            "abstract": (
                "Standard single-cell RNA-sequencing analysis (scRNA-seq) workflows consist of converting raw read data into cell-gene count matrices through sequence alignment, followed by analyses including filtering, highly variable gene selection, dimensionality reduction, clustering, and differential expression analysis. Seurat and Scanpy are the most widely-used packages implementing such workflows, and are generally thought to implement individual steps similarly. We investigate in detail the algorithms and methods underlying Seurat and Scanpy and find that there are, in fact, considerable differences in the outputs of Seurat and Scanpy. The extent of differences between the programs is approximately equivalent to the variability that would be introduced in benchmarking scRNA-seq datasets by sequencing less than 5% of the reads or analyzing less than 20% of the cell population. Additionally, distinct versions of Seurat and Scanpy can produce very different results, especially during parts of differential expression analysis. Our analysis highlights the need for users of scRNA-seq to carefully assess the tools on which they rely, and the importance of developers of scientific software to prioritize transparency, consistency, and reproducibility for their tools."
            ),
            "link": "https://doi.org/10.1101/2024.04.04.588111",
            "date": "11-April-2024",
            "expected": {
                "name": None,
                "num_images": None,
                "num_patients": None,
                "modalities": None,
                "body_regions": None,
                "additional_data": None,
            },
            "name_aliases": None,
        },
    }
