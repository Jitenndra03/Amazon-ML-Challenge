import pandas as pd
import numpy as np
import re

df_test=pd.read_csv('dataset/test.csv')

def create_features(df):
    df_eng=df.copy()
    if 'price' in df_eng.columns:
        df_eng['log_price']=np.log1p(df_eng['price'])
    
    boilerplate_words=['point','bullet','value','item','unit','description','product']
    clean_text=df_eng['catalog_content'].copy()
    for word in boilerplate_words:
        clean_text=clean_text.str.replace(r'\b{}\b'.format(word),'',regex=True,flags=re.IGNORECASE)
    df_eng['catalog_content_cleaned']=clean_text

    def extract_ipq(text):
        match=re.search(r'(\d+)\s*(?:count|pack|pc|pcs|set)',text,re.IGNORECASE)
        return int(match.group(1)) if match else 1
    df_eng['ipq']=df_eng['catalog_content_cleaned'].apply(extract_ipq)

    def extract_quantity_and_unit(text):
        pattern=r'(\d+\.?\d*)\s*(oz|ounce|ounces|lb|lbs|pound|pounds|g|gram|grams|kg|kilo|kilogram|fl oz|floz|l|liter|ml|milliliter|gallon|gallons)'
        match=re.search(pattern,text,re.IGNORECASE)
        return pd.Series([float(match.group(1)),match.group(2).lower()]) if match else pd.Series([np.nan,np.nan])
    df_eng[['quantity','unit']]=df_eng['catalog_content_cleaned'].apply(extract_quantity_and_unit)
    
    unit_map={'ounce':'oz','ounces':'oz','lb':'lb','lbs':'lb','pound':'lb','pounds':'lb','g':'g','gram':'g','grams':'g','kg':'kg','kilo':'kg','kilogram':'kg','fl oz':'fl oz','floz':'fl oz','l':'l','liter':'l','ml':'ml','milliliter':'ml'}
    df_eng['unit_cleaned']=df_eng['unit'].map(unit_map).fillna(df_eng['unit'])
    
    df_eng['is_organic']=df_eng['catalog_content_cleaned'].str.contains('organic',case=False,na=False)
    df_eng['is_gluten_free']=df_eng['catalog_content_cleaned'].str.contains('gluten free',case=False,na=False)
    df_eng['is_kosher']=df_eng['catalog_content_cleaned'].str.contains('kosher',case=False,na=False)
    
    def extract_item_title(text):
        match=re.search(r'Name:\s*(.*?)(,|$)',text,re.IGNORECASE)
        return match.group(1).strip() if match and match.group(1).strip() else "Unknown"
    df_eng['Item Name']=df_eng['catalog_content_cleaned'].apply(extract_item_title)
    
    def extract_brand(title):
        if title and title!="Unknown": return title.split()[0]
        return "Unknown"
    df_eng['brand']=df_eng['Item Name'].apply(extract_brand)
    
    # We also need these columns for the embedding and prediction steps
    df_eng['image_link'] = df['image_link']
    
    return df_eng

df_processed=create_features(df_test)

final_columns=[
    'sample_id',
    'brand',
    'ipq',
    'quantity',
    'unit_cleaned',
    'is_organic',
    'is_gluten_free',
    'is_kosher',
    'Item Name',
    'image_link', # Keeping this for the embedding step
    'catalog_content_cleaned' # Keeping this for the embedding step
]
df_to_save=df_processed.reindex(columns=final_columns).copy()

df_to_save['quantity']=df_to_save['quantity'].fillna(0)
df_to_save['unit_cleaned']=df_to_save['unit_cleaned'].fillna('unknown')

bool_cols=df_to_save.select_dtypes(include='bool').columns
df_to_save[bool_cols]=df_to_save[bool_cols].astype(int)

output_filename='feature_test_model_ready.csv'
df_to_save.to_csv(output_filename,index=False)

print(f"Final file '{output_filename}' created successfully with the requested columns.")
print(df_to_save.info())