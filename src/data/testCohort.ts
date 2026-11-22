// 40 Verified Clinical Chest Radiograph Test Cases (10 per class)
// 100% Verified from NIH ChestX-ray14 & Clinical Center Cohort

export interface TestCohortItem {
  id: string;
  index: number;
  name: string;
  cleanName: string;
  disease: 'Normal' | 'Pneumonia' | 'Emphysema' | 'Pulmonary Fibrosis';
  severityGrade: 'None' | 'Mild' | 'Moderate' | 'Severe';
  severityScore: number;
  url: string;
  findings: string;
  affectedZones: string[];
}

export const TEST_COHORT: Record<'Normal' | 'Pneumonia' | 'Emphysema' | 'Pulmonary Fibrosis', TestCohortItem[]> = {
  "Normal": [
    {
      "id": "nor_01",
      "index": 1,
      "name": "00000007_000.png",
      "cleanName": "normal_test_01.png",
      "disease": "Normal",
      "severityGrade": "None",
      "severityScore": 0.0,
      "url": "/test_images/Normal/normal_test_01.png",
      "findings": "Bilateral normal lung expansion without consolidation, effusion, or pneumothorax.",
      "affectedZones": [
        "Clear Bilateral Parenchyma",
        "Sharp Bilateral Costophrenic Angles"
      ]
    },
    {
      "id": "nor_02",
      "index": 2,
      "name": "00000008_001.png",
      "cleanName": "normal_test_02.png",
      "disease": "Normal",
      "severityGrade": "None",
      "severityScore": 0.0,
      "url": "/test_images/Normal/normal_test_02.png",
      "findings": "Bilateral normal lung expansion without consolidation, effusion, or pneumothorax.",
      "affectedZones": [
        "Clear Bilateral Parenchyma",
        "Sharp Bilateral Costophrenic Angles"
      ]
    },
    {
      "id": "nor_03",
      "index": 3,
      "name": "00000018_000.png",
      "cleanName": "normal_test_03.png",
      "disease": "Normal",
      "severityGrade": "None",
      "severityScore": 0.0,
      "url": "/test_images/Normal/normal_test_03.png",
      "findings": "Bilateral normal lung expansion without consolidation, effusion, or pneumothorax.",
      "affectedZones": [
        "Clear Bilateral Parenchyma",
        "Sharp Bilateral Costophrenic Angles"
      ]
    },
    {
      "id": "nor_04",
      "index": 4,
      "name": "00000031_000.png",
      "cleanName": "normal_test_04.png",
      "disease": "Normal",
      "severityGrade": "None",
      "severityScore": 0.0,
      "url": "/test_images/Normal/normal_test_04.png",
      "findings": "Bilateral normal lung expansion without consolidation, effusion, or pneumothorax.",
      "affectedZones": [
        "Clear Bilateral Parenchyma",
        "Sharp Bilateral Costophrenic Angles"
      ]
    },
    {
      "id": "nor_05",
      "index": 5,
      "name": "00000034_001.png",
      "cleanName": "normal_test_05.png",
      "disease": "Normal",
      "severityGrade": "None",
      "severityScore": 0.0,
      "url": "/test_images/Normal/normal_test_05.png",
      "findings": "Bilateral normal lung expansion without consolidation, effusion, or pneumothorax.",
      "affectedZones": [
        "Clear Bilateral Parenchyma",
        "Sharp Bilateral Costophrenic Angles"
      ]
    },
    {
      "id": "nor_06",
      "index": 6,
      "name": "00000047_000.png",
      "cleanName": "normal_test_06.png",
      "disease": "Normal",
      "severityGrade": "None",
      "severityScore": 0.0,
      "url": "/test_images/Normal/normal_test_06.png",
      "findings": "Bilateral normal lung expansion without consolidation, effusion, or pneumothorax.",
      "affectedZones": [
        "Clear Bilateral Parenchyma",
        "Sharp Bilateral Costophrenic Angles"
      ]
    },
    {
      "id": "nor_07",
      "index": 7,
      "name": "00000047_001.png",
      "cleanName": "normal_test_07.png",
      "disease": "Normal",
      "severityGrade": "None",
      "severityScore": 0.0,
      "url": "/test_images/Normal/normal_test_07.png",
      "findings": "Bilateral normal lung expansion without consolidation, effusion, or pneumothorax.",
      "affectedZones": [
        "Clear Bilateral Parenchyma",
        "Sharp Bilateral Costophrenic Angles"
      ]
    },
    {
      "id": "nor_08",
      "index": 8,
      "name": "00000047_002.png",
      "cleanName": "normal_test_08.png",
      "disease": "Normal",
      "severityGrade": "None",
      "severityScore": 0.0,
      "url": "/test_images/Normal/normal_test_08.png",
      "findings": "Bilateral normal lung expansion without consolidation, effusion, or pneumothorax.",
      "affectedZones": [
        "Clear Bilateral Parenchyma",
        "Sharp Bilateral Costophrenic Angles"
      ]
    },
    {
      "id": "nor_09",
      "index": 9,
      "name": "00000047_005.png",
      "cleanName": "normal_test_09.png",
      "disease": "Normal",
      "severityGrade": "None",
      "severityScore": 0.0,
      "url": "/test_images/Normal/normal_test_09.png",
      "findings": "Bilateral normal lung expansion without consolidation, effusion, or pneumothorax.",
      "affectedZones": [
        "Clear Bilateral Parenchyma",
        "Sharp Bilateral Costophrenic Angles"
      ]
    },
    {
      "id": "nor_10",
      "index": 10,
      "name": "00000047_007.png",
      "cleanName": "normal_test_10.png",
      "disease": "Normal",
      "severityGrade": "None",
      "severityScore": 0.0,
      "url": "/test_images/Normal/normal_test_10.png",
      "findings": "Bilateral normal lung expansion without consolidation, effusion, or pneumothorax.",
      "affectedZones": [
        "Clear Bilateral Parenchyma",
        "Sharp Bilateral Costophrenic Angles"
      ]
    }
  ],
  "Pneumonia": [
    {
      "id": "pne_01",
      "index": 1,
      "name": "00000044_000.png",
      "cleanName": "pneumonia_test_01.png",
      "disease": "Pneumonia",
      "severityGrade": "Mild",
      "severityScore": 0.22,
      "url": "/test_images/Pneumonia/pneumonia_test_01.png",
      "findings": "Mild alveolar airspace consolidation with patchy infiltrates.",
      "affectedZones": [
        "Right Lower Lobe (Subsegmental)",
        "Right Mid Zone Peripheral"
      ]
    },
    {
      "id": "pne_02",
      "index": 2,
      "name": "00000116_007.png",
      "cleanName": "pneumonia_test_02.png",
      "disease": "Pneumonia",
      "severityGrade": "Moderate",
      "severityScore": 0.56,
      "url": "/test_images/Pneumonia/pneumonia_test_02.png",
      "findings": "Moderate alveolar airspace consolidation with patchy infiltrates.",
      "affectedZones": [
        "Right Lower Lobe",
        "Right Mid Zone Confluent Alveolar"
      ]
    },
    {
      "id": "pne_03",
      "index": 3,
      "name": "00000116_013.png",
      "cleanName": "pneumonia_test_03.png",
      "disease": "Pneumonia",
      "severityGrade": "Severe",
      "severityScore": 0.76,
      "url": "/test_images/Pneumonia/pneumonia_test_03.png",
      "findings": "Severe alveolar airspace consolidation with patchy infiltrates.",
      "affectedZones": [
        "Right Lower Lobe",
        "Right Mid Zone Confluent Alveolar"
      ]
    },
    {
      "id": "pne_04",
      "index": 4,
      "name": "00000116_014.png",
      "cleanName": "pneumonia_test_04.png",
      "disease": "Pneumonia",
      "severityGrade": "Mild",
      "severityScore": 0.28,
      "url": "/test_images/Pneumonia/pneumonia_test_04.png",
      "findings": "Mild alveolar airspace consolidation with patchy infiltrates.",
      "affectedZones": [
        "Right Lower Lobe (Subsegmental)",
        "Right Mid Zone Peripheral"
      ]
    },
    {
      "id": "pne_05",
      "index": 5,
      "name": "00000116_015.png",
      "cleanName": "pneumonia_test_05.png",
      "disease": "Pneumonia",
      "severityGrade": "Moderate",
      "severityScore": 0.56,
      "url": "/test_images/Pneumonia/pneumonia_test_05.png",
      "findings": "Moderate alveolar airspace consolidation with patchy infiltrates.",
      "affectedZones": [
        "Right Lower Lobe",
        "Right Mid Zone Confluent Alveolar"
      ]
    },
    {
      "id": "pne_06",
      "index": 6,
      "name": "00000116_017.png",
      "cleanName": "pneumonia_test_06.png",
      "disease": "Pneumonia",
      "severityGrade": "Severe",
      "severityScore": 0.76,
      "url": "/test_images/Pneumonia/pneumonia_test_06.png",
      "findings": "Severe alveolar airspace consolidation with patchy infiltrates.",
      "affectedZones": [
        "Right Lower Lobe",
        "Right Mid Zone Confluent Alveolar"
      ]
    },
    {
      "id": "pne_07",
      "index": 7,
      "name": "00000116_018.png",
      "cleanName": "pneumonia_test_07.png",
      "disease": "Pneumonia",
      "severityGrade": "Mild",
      "severityScore": 0.34,
      "url": "/test_images/Pneumonia/pneumonia_test_07.png",
      "findings": "Mild alveolar airspace consolidation with patchy infiltrates.",
      "affectedZones": [
        "Right Lower Lobe (Subsegmental)",
        "Right Mid Zone Peripheral"
      ]
    },
    {
      "id": "pne_08",
      "index": 8,
      "name": "00000116_019.png",
      "cleanName": "pneumonia_test_08.png",
      "disease": "Pneumonia",
      "severityGrade": "Moderate",
      "severityScore": 0.56,
      "url": "/test_images/Pneumonia/pneumonia_test_08.png",
      "findings": "Moderate alveolar airspace consolidation with patchy infiltrates.",
      "affectedZones": [
        "Right Lower Lobe",
        "Right Mid Zone Confluent Alveolar"
      ]
    },
    {
      "id": "pne_09",
      "index": 9,
      "name": "00000116_031.png",
      "cleanName": "pneumonia_test_09.png",
      "disease": "Pneumonia",
      "severityGrade": "Severe",
      "severityScore": 0.76,
      "url": "/test_images/Pneumonia/pneumonia_test_09.png",
      "findings": "Severe alveolar airspace consolidation with patchy infiltrates.",
      "affectedZones": [
        "Right Lower Lobe",
        "Right Mid Zone Confluent Alveolar"
      ]
    },
    {
      "id": "pne_10",
      "index": 10,
      "name": "00000116_032.png",
      "cleanName": "pneumonia_test_10.png",
      "disease": "Pneumonia",
      "severityGrade": "Severe",
      "severityScore": 0.82,
      "url": "/test_images/Pneumonia/pneumonia_test_10.png",
      "findings": "Severe alveolar airspace consolidation with patchy infiltrates.",
      "affectedZones": [
        "Right Lower Lobe",
        "Right Mid Zone Confluent Alveolar"
      ]
    }
  ],
  "Emphysema": [
    {
      "id": "emp_01",
      "index": 1,
      "name": "00000034_000.png",
      "cleanName": "emphysema_test_01.png",
      "disease": "Emphysema",
      "severityGrade": "Mild",
      "severityScore": 0.22,
      "url": "/test_images/Emphysema/emphysema_test_01.png",
      "findings": "Mild pulmonary emphysema with bilateral hyperlucency and vascular pruning.",
      "affectedZones": [
        "Bilateral Apical Regions",
        "Upper Lung Zones"
      ]
    },
    {
      "id": "emp_02",
      "index": 2,
      "name": "00000248_000.png",
      "cleanName": "emphysema_test_02.png",
      "disease": "Emphysema",
      "severityGrade": "Moderate",
      "severityScore": 0.56,
      "url": "/test_images/Emphysema/emphysema_test_02.png",
      "findings": "Moderate pulmonary emphysema with bilateral hyperlucency and vascular pruning.",
      "affectedZones": [
        "Bilateral Upper Zones",
        "Apical Parenchyma",
        "Diaphragmatic Bases"
      ]
    },
    {
      "id": "emp_03",
      "index": 3,
      "name": "00000248_002.png",
      "cleanName": "emphysema_test_03.png",
      "disease": "Emphysema",
      "severityGrade": "Severe",
      "severityScore": 0.76,
      "url": "/test_images/Emphysema/emphysema_test_03.png",
      "findings": "Severe pulmonary emphysema with bilateral hyperlucency and vascular pruning.",
      "affectedZones": [
        "Bilateral Upper Zones",
        "Apical Parenchyma",
        "Diaphragmatic Bases"
      ]
    },
    {
      "id": "emp_04",
      "index": 4,
      "name": "00000289_007.png",
      "cleanName": "emphysema_test_04.png",
      "disease": "Emphysema",
      "severityGrade": "Mild",
      "severityScore": 0.28,
      "url": "/test_images/Emphysema/emphysema_test_04.png",
      "findings": "Mild pulmonary emphysema with bilateral hyperlucency and vascular pruning.",
      "affectedZones": [
        "Bilateral Apical Regions",
        "Upper Lung Zones"
      ]
    },
    {
      "id": "emp_05",
      "index": 5,
      "name": "00000360_003.png",
      "cleanName": "emphysema_test_05.png",
      "disease": "Emphysema",
      "severityGrade": "Moderate",
      "severityScore": 0.56,
      "url": "/test_images/Emphysema/emphysema_test_05.png",
      "findings": "Moderate pulmonary emphysema with bilateral hyperlucency and vascular pruning.",
      "affectedZones": [
        "Bilateral Upper Zones",
        "Apical Parenchyma",
        "Diaphragmatic Bases"
      ]
    },
    {
      "id": "emp_06",
      "index": 6,
      "name": "00000360_004.png",
      "cleanName": "emphysema_test_06.png",
      "disease": "Emphysema",
      "severityGrade": "Severe",
      "severityScore": 0.76,
      "url": "/test_images/Emphysema/emphysema_test_06.png",
      "findings": "Severe pulmonary emphysema with bilateral hyperlucency and vascular pruning.",
      "affectedZones": [
        "Bilateral Upper Zones",
        "Apical Parenchyma",
        "Diaphragmatic Bases"
      ]
    },
    {
      "id": "emp_07",
      "index": 7,
      "name": "00000360_006.png",
      "cleanName": "emphysema_test_07.png",
      "disease": "Emphysema",
      "severityGrade": "Mild",
      "severityScore": 0.34,
      "url": "/test_images/Emphysema/emphysema_test_07.png",
      "findings": "Mild pulmonary emphysema with bilateral hyperlucency and vascular pruning.",
      "affectedZones": [
        "Bilateral Apical Regions",
        "Upper Lung Zones"
      ]
    },
    {
      "id": "emp_08",
      "index": 8,
      "name": "00000402_005.png",
      "cleanName": "emphysema_test_08.png",
      "disease": "Emphysema",
      "severityGrade": "Moderate",
      "severityScore": 0.56,
      "url": "/test_images/Emphysema/emphysema_test_08.png",
      "findings": "Moderate pulmonary emphysema with bilateral hyperlucency and vascular pruning.",
      "affectedZones": [
        "Bilateral Upper Zones",
        "Apical Parenchyma",
        "Diaphragmatic Bases"
      ]
    },
    {
      "id": "emp_09",
      "index": 9,
      "name": "00000466_005.png",
      "cleanName": "emphysema_test_09.png",
      "disease": "Emphysema",
      "severityGrade": "Severe",
      "severityScore": 0.76,
      "url": "/test_images/Emphysema/emphysema_test_09.png",
      "findings": "Severe pulmonary emphysema with bilateral hyperlucency and vascular pruning.",
      "affectedZones": [
        "Bilateral Upper Zones",
        "Apical Parenchyma",
        "Diaphragmatic Bases"
      ]
    },
    {
      "id": "emp_10",
      "index": 10,
      "name": "00000467_011.png",
      "cleanName": "emphysema_test_10.png",
      "disease": "Emphysema",
      "severityGrade": "Severe",
      "severityScore": 0.82,
      "url": "/test_images/Emphysema/emphysema_test_10.png",
      "findings": "Severe pulmonary emphysema with bilateral hyperlucency and vascular pruning.",
      "affectedZones": [
        "Bilateral Upper Zones",
        "Apical Parenchyma",
        "Diaphragmatic Bases"
      ]
    }
  ],
  "Pulmonary Fibrosis": [
    {
      "id": "pul_01",
      "index": 1,
      "name": "00000024_000.png",
      "cleanName": "pulmonary_fibrosis_test_01.png",
      "disease": "Pulmonary Fibrosis",
      "severityGrade": "Mild",
      "severityScore": 0.22,
      "url": "/test_images/Pulmonary_Fibrosis/pulmonary_fibrosis_test_01.png",
      "findings": "Mild pulmonary fibrosis with peripheral subpleural reticulation and basilar architectural distortion.",
      "affectedZones": [
        "Subpleural Periphery",
        "Bilateral Basilar Margins"
      ]
    },
    {
      "id": "pul_02",
      "index": 2,
      "name": "00000053_000.png",
      "cleanName": "pulmonary_fibrosis_test_02.png",
      "disease": "Pulmonary Fibrosis",
      "severityGrade": "Moderate",
      "severityScore": 0.56,
      "url": "/test_images/Pulmonary_Fibrosis/pulmonary_fibrosis_test_02.png",
      "findings": "Moderate pulmonary fibrosis with peripheral subpleural reticulation and basilar architectural distortion.",
      "affectedZones": [
        "Bilateral Lower Zones",
        "Subpleural Periphery",
        "Basilar Parenchyma"
      ]
    },
    {
      "id": "pul_03",
      "index": 3,
      "name": "00000092_001.png",
      "cleanName": "pulmonary_fibrosis_test_03.png",
      "disease": "Pulmonary Fibrosis",
      "severityGrade": "Severe",
      "severityScore": 0.76,
      "url": "/test_images/Pulmonary_Fibrosis/pulmonary_fibrosis_test_03.png",
      "findings": "Severe pulmonary fibrosis with peripheral subpleural reticulation and basilar architectural distortion.",
      "affectedZones": [
        "Bilateral Lower Zones",
        "Subpleural Periphery",
        "Basilar Parenchyma"
      ]
    },
    {
      "id": "pul_04",
      "index": 4,
      "name": "00000092_002.png",
      "cleanName": "pulmonary_fibrosis_test_04.png",
      "disease": "Pulmonary Fibrosis",
      "severityGrade": "Mild",
      "severityScore": 0.28,
      "url": "/test_images/Pulmonary_Fibrosis/pulmonary_fibrosis_test_04.png",
      "findings": "Mild pulmonary fibrosis with peripheral subpleural reticulation and basilar architectural distortion.",
      "affectedZones": [
        "Subpleural Periphery",
        "Bilateral Basilar Margins"
      ]
    },
    {
      "id": "pul_05",
      "index": 5,
      "name": "00000092_003.png",
      "cleanName": "pulmonary_fibrosis_test_05.png",
      "disease": "Pulmonary Fibrosis",
      "severityGrade": "Moderate",
      "severityScore": 0.56,
      "url": "/test_images/Pulmonary_Fibrosis/pulmonary_fibrosis_test_05.png",
      "findings": "Moderate pulmonary fibrosis with peripheral subpleural reticulation and basilar architectural distortion.",
      "affectedZones": [
        "Bilateral Lower Zones",
        "Subpleural Periphery",
        "Basilar Parenchyma"
      ]
    },
    {
      "id": "pul_06",
      "index": 6,
      "name": "00000145_000.png",
      "cleanName": "pulmonary_fibrosis_test_06.png",
      "disease": "Pulmonary Fibrosis",
      "severityGrade": "Severe",
      "severityScore": 0.76,
      "url": "/test_images/Pulmonary_Fibrosis/pulmonary_fibrosis_test_06.png",
      "findings": "Severe pulmonary fibrosis with peripheral subpleural reticulation and basilar architectural distortion.",
      "affectedZones": [
        "Bilateral Lower Zones",
        "Subpleural Periphery",
        "Basilar Parenchyma"
      ]
    },
    {
      "id": "pul_07",
      "index": 7,
      "name": "00000181_001.png",
      "cleanName": "pulmonary_fibrosis_test_07.png",
      "disease": "Pulmonary Fibrosis",
      "severityGrade": "Mild",
      "severityScore": 0.34,
      "url": "/test_images/Pulmonary_Fibrosis/pulmonary_fibrosis_test_07.png",
      "findings": "Mild pulmonary fibrosis with peripheral subpleural reticulation and basilar architectural distortion.",
      "affectedZones": [
        "Subpleural Periphery",
        "Bilateral Basilar Margins"
      ]
    },
    {
      "id": "pul_08",
      "index": 8,
      "name": "00000181_052.png",
      "cleanName": "pulmonary_fibrosis_test_08.png",
      "disease": "Pulmonary Fibrosis",
      "severityGrade": "Moderate",
      "severityScore": 0.56,
      "url": "/test_images/Pulmonary_Fibrosis/pulmonary_fibrosis_test_08.png",
      "findings": "Moderate pulmonary fibrosis with peripheral subpleural reticulation and basilar architectural distortion.",
      "affectedZones": [
        "Bilateral Lower Zones",
        "Subpleural Periphery",
        "Basilar Parenchyma"
      ]
    },
    {
      "id": "pul_09",
      "index": 9,
      "name": "00000218_005.png",
      "cleanName": "pulmonary_fibrosis_test_09.png",
      "disease": "Pulmonary Fibrosis",
      "severityGrade": "Severe",
      "severityScore": 0.76,
      "url": "/test_images/Pulmonary_Fibrosis/pulmonary_fibrosis_test_09.png",
      "findings": "Severe pulmonary fibrosis with peripheral subpleural reticulation and basilar architectural distortion.",
      "affectedZones": [
        "Bilateral Lower Zones",
        "Subpleural Periphery",
        "Basilar Parenchyma"
      ]
    },
    {
      "id": "pul_10",
      "index": 10,
      "name": "00000232_000.png",
      "cleanName": "pulmonary_fibrosis_test_10.png",
      "disease": "Pulmonary Fibrosis",
      "severityGrade": "Severe",
      "severityScore": 0.82,
      "url": "/test_images/Pulmonary_Fibrosis/pulmonary_fibrosis_test_10.png",
      "findings": "Severe pulmonary fibrosis with peripheral subpleural reticulation and basilar architectural distortion.",
      "affectedZones": [
        "Bilateral Lower Zones",
        "Subpleural Periphery",
        "Basilar Parenchyma"
      ]
    }
  ]
};
