import pandas as pd

# Load the dataset with the correct encoding
file_path = '臺北市主要死亡原因(97至111年).csv'
try:
    dataset = pd.read_csv(file_path, encoding='big5')
except UnicodeDecodeError:
    print("Error loading file: The file is not in Big5 encoding.")
    # Additional error handling can be implemented here

# Translation dictionary, including column names
translation_dict = {
    '年份': 'Year',
    '性別': 'Gender',
    '疾病死因': 'Cause of Death',
    '順位': 'Rank',
    '死亡人數': 'Number of Deaths',
    '死亡率': 'Death Rate',
    '標準化死亡率': 'Standardized Death Rate',
    '死亡人數結構比': 'Proportion of Deaths',
    '總計': 'Total',
    '所有死因': 'All Causes of Death',
    '惡性腫瘤': 'Malignant Neoplasm',
    '心臟疾病（高血壓性疾病除外）': 'Heart Disease (excluding hypertensive disease)',
    '腦血管疾病': 'Cerebrovascular Disease',
    '肺炎': 'Pneumonia',
    '胃及十二指腸潰瘍': 'Gastric and Duodenal Ulcer',
    '腎炎、腎病症候群及腎病變': 'Nephritis, Nephrotic Syndrome, and Nephrosis',
    '腦膜炎': 'Meningitis',
    '腸道感染症': 'Intestinal Infectious Diseases',
    '膽結石及其他膽囊疾患': 'Gallstones and Other Gallbladder Diseases',
    '蓄意自我傷害（自殺）': 'Intentional Self-Harm (Suicide)',
    '血管性及未明示之癡呆症': 'Vascular and Unspecified Dementia',
    '衰老/老邁': 'Senility/Aging',
    '貧血': 'Anemia',
    '阿茲海默病': "Alzheimer's Disease",
    '骨骼肌肉系統及結締組織之疾病': 'Diseases of the Musculoskeletal System and Connective Tissue',
    '高血壓性疾病': 'Hypertensive Diseases',
    '塵肺症': 'Pneumoconiosis',
    '嬰兒猝死症候群（SIDS）': 'Sudden Infant Death Syndrome (SIDS)',
    '動脈粥樣硬化': 'Atherosclerosis',
    '妊娠(懷孕)、生產及產褥期': 'Pregnancy, Childbirth and the Puerperium',
    '急性支氣管炎及急性細支氣管炎': 'Acute Bronchitis and Acute Bronchiolitis',
    '嚴重特殊傳染性肺炎（COVID-19）': 'Severe Acute Respiratory Syndrome (COVID-19)',
    '其他': 'Others',
    '主動脈瘤及剝離': 'Aortic Aneurysm and Dissection',
    '事故傷害': 'Accidental Injury',
    '人類免疫缺乏病毒（HIV）疾病': 'Human Immunodeficiency Virus (HIV) Disease',
    '先天性畸形變形及染色體異常': 'Congenital Malformations, Deformations, and Chromosomal Abnormalities',
    '加害（他殺）': 'Homicide',
    '原位與良性腫瘤（惡性腫瘤除外）': 'In Situ and Benign Tumors (excluding Malignant Neoplasms)',
    '慢性下呼吸道疾病': 'Chronic Lower Respiratory Diseases',
    '慢性肝病及肝硬化': 'Chronic Liver Disease and Cirrhosis',
    '敗血症': 'Sepsis',
    '椎骨肌肉萎縮及有關聯之症候群': 'Spinal Muscular Atrophy and Related Syndromes',
    '流行性感冒': 'Influenza',
    '源於周產期的特定病況': 'Conditions Originating in the Perinatal Period',
    '疝氣及腸阻塞': 'Hernia and Intestinal Obstruction',
    '病毒性肝炎': 'Viral Hepatitis',
    '皮膚及皮下組織疾病': 'Diseases of the Skin and Subcutaneous Tissue',
    '結核病': 'Tuberculosis',
    '肇因於吸入外物之肺部病況（塵肺症及肺炎除外）': 'Conditions of the Lung due to Inhalation of External Substances (excluding Pneumoconiosis and Pneumonia)',
    '女性': 'Female',
    '帕金森病': "Parkinson's Disease",
    '男性': 'Male',
    '糖尿病': 'Diabetes'
}

# Function to translate text using the dictionary
def translate_text(text):
    return f"{translation_dict.get(text, text)} ({text})" if text in translation_dict else text

# Function to convert Taiwanese year to Gregorian year
def convert_year(taiwanese_year):
    try:
        gregorian_year = int(taiwanese_year.replace('年', '')) + 1911
        return f"{gregorian_year} ({taiwanese_year})"
    except ValueError:
        return taiwanese_year

# Creating a new DataFrame for translated data
translated_dataset = dataset.copy()

# Translate the dataset
for col in translated_dataset.columns:
    if col == '年份':
        translated_dataset[col] = translated_dataset[col].apply(convert_year)
    elif translated_dataset[col].dtype == object:
        translated_dataset[col] = translated_dataset[col].apply(translate_text)

# Translate column names after translating data
translated_columns = {col: translate_text(col) for col in translated_dataset.columns}
translated_dataset.rename(columns=translated_columns, inplace=True)

# Save the original dataset and the translated dataset as XLSX
xlsx_file_path = 'translated_dataset.xlsx'  # Output file path
with pd.ExcelWriter(xlsx_file_path, engine='xlsxwriter') as writer:
    dataset.to_excel(writer, index=False, sheet_name='Original Data')
    translated_dataset.to_excel(writer, index=False, sheet_name='Translated Data')
