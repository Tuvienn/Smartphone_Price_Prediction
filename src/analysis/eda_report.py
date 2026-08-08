import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

def run_eda():
    df = pd.read_csv('data/processed/fptshop_smartphones_clean.csv')
    plot_dir = 'data/processed/plots'
    os.makedirs(plot_dir, exist_ok=True)
    
    # Setup plotting style
    sns.set_theme(style="whitegrid")
    
    print("--- EDA REPORT ---")
    print(f"Dataset shape: {df.shape}")
    print("\nDescriptive Statistics (Numeric):")
    print(df.describe())
    
    # 1. Price histogram
    plt.figure(figsize=(10, 6))
    sns.histplot(df['price_vnd'], kde=True, bins=30)
    plt.title('Distribution of Smartphone Prices')
    plt.xlabel('Price (VNĐ)')
    plt.ylabel('Count')
    plt.savefig(f'{plot_dir}/01_price_histogram.png')
    plt.close()
    
    # 2. Price boxplot
    plt.figure(figsize=(10, 4))
    sns.boxplot(x=df['price_vnd'])
    plt.title('Boxplot of Smartphone Prices')
    plt.xlabel('Price (VNĐ)')
    plt.savefig(f'{plot_dir}/02_price_boxplot.png')
    plt.close()
    
    # 3. Brand count plot
    plt.figure(figsize=(12, 6))
    order = df['brand'].value_counts().index
    sns.countplot(y=df['brand'], order=order, hue=df['brand'], legend=False)
    plt.title('Number of Smartphones by Brand')
    plt.xlabel('Count')
    plt.ylabel('Brand')
    plt.savefig(f'{plot_dir}/03_brand_count.png')
    plt.close()
    
    # 4. Price by brand
    plt.figure(figsize=(12, 6))
    sns.boxplot(x='price_vnd', y='brand', data=df, order=order, hue='brand', legend=False)
    plt.title('Price Distribution by Brand')
    plt.xlabel('Price (VNĐ)')
    plt.ylabel('Brand')
    plt.savefig(f'{plot_dir}/04_price_by_brand.png')
    plt.close()
    
    # 5. RAM distribution
    plt.figure(figsize=(10, 6))
    sns.countplot(x='ram_gb', data=df, hue='ram_gb', legend=False)
    plt.title('Distribution of RAM Capacity')
    plt.xlabel('RAM (GB)')
    plt.ylabel('Count')
    plt.savefig(f'{plot_dir}/05_ram_distribution.png')
    plt.close()
    
    # 6. Price vs RAM
    plt.figure(figsize=(10, 6))
    sns.boxplot(x='ram_gb', y='price_vnd', data=df, hue='ram_gb', legend=False)
    plt.title('Price by RAM Capacity')
    plt.xlabel('RAM (GB)')
    plt.ylabel('Price (VNĐ)')
    plt.savefig(f'{plot_dir}/06_price_vs_ram.png')
    plt.close()
    
    # 7. Storage distribution
    plt.figure(figsize=(10, 6))
    sns.countplot(x='storage_gb', data=df, hue='storage_gb', legend=False)
    plt.title('Distribution of Storage Capacity')
    plt.xlabel('Storage (GB)')
    plt.ylabel('Count')
    plt.savefig(f'{plot_dir}/07_storage_distribution.png')
    plt.close()
    
    # 8. Price vs Storage
    plt.figure(figsize=(10, 6))
    sns.boxplot(x='storage_gb', y='price_vnd', data=df, hue='storage_gb', legend=False)
    plt.title('Price by Storage Capacity')
    plt.xlabel('Storage (GB)')
    plt.ylabel('Price (VNĐ)')
    plt.savefig(f'{plot_dir}/08_price_vs_storage.png')
    plt.close()
    
    # 9. Price vs Screen Size
    plt.figure(figsize=(10, 6))
    sns.scatterplot(x='screen_size_inch', y='price_vnd', data=df, hue='brand')
    plt.title('Price vs Screen Size')
    plt.xlabel('Screen Size (inch)')
    plt.ylabel('Price (VNĐ)')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize='small')
    plt.tight_layout()
    plt.savefig(f'{plot_dir}/09_price_vs_screen.png')
    plt.close()
    
    # 10. Correlation heatmap
    plt.figure(figsize=(8, 6))
    numeric_df = df[['ram_gb', 'storage_gb', 'screen_size_inch', 'main_camera_mp', 'price_vnd']]
    corr = numeric_df.corr()
    sns.heatmap(corr, annot=True, cmap='coolwarm', vmin=-1, vmax=1, fmt=".2f")
    plt.title('Correlation Matrix of Numeric Features')
    plt.tight_layout()
    plt.savefig(f'{plot_dir}/10_correlation_heatmap.png')
    plt.close()
    
    # Additional: 5G vs Price (Since 5G has 37% missing, we want to see if there's a difference where it exists)
    df_5g = df.dropna(subset=['has_5g'])
    if not df_5g.empty:
        plt.figure(figsize=(8, 6))
        sns.boxplot(x='has_5g', y='price_vnd', data=df_5g, hue='has_5g', legend=False)
        plt.title('Price by 5G Support (where known)')
        plt.xlabel('Has 5G')
        plt.ylabel('Price (VNĐ)')
        plt.savefig(f'{plot_dir}/11_price_vs_5g.png')
        plt.close()
        
    print("\nEDA script completed successfully. Plots saved to data/processed/plots/")

if __name__ == "__main__":
    run_eda()
