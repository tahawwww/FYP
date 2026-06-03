import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

def save_performance_table(results):
    df = pd.DataFrame(results)
    print("\n" + "="*50)
    print("FINAL RESEARCH PERFORMANCE ANALYSIS")
    print("="*50)
    print(df[['Test', 'Avg AoI', 'Avg AoLI']])
    df.to_csv("results/performance_metrics.csv")

def plot_aoi_v_aoli(results):
    df = pd.DataFrame(results)
    
    # Setting 'Test' as the index tells Pandas to use it for the X-axis
    df.set_index('Test', inplace=True)
    
    # Pandas automatically groups columns side-by-side when plotting!
    ax = df[['Avg AoI', 'Avg AoLI']].plot(
        kind='bar', 
        figsize=(10, 6), 
        rot=0, # Keeps the text horizontal instead of 45 degrees
        color=['#1f77b4', '#ff7f0e'] # Clean blue and orange
    )
    
    # Formatting aesthetics
    plt.title('6G HDRL: Security vs. Freshness Tradeoff', fontsize=14)
    plt.ylabel('Time Steps', fontsize=12)
    plt.xlabel('') # Hides the redundant "Test" label on the x-axis
    plt.legend(['Freshness (AoI)', 'Secrecy (AoLI)'])
    
    # Automatically attach the exact floating-point numbers to the top of every bar
    for container in ax.containers:
        ax.bar_label(container, fmt='%.2f', padding=3)
        
    # Give the chart a little extra headroom so the text doesn't hit the ceiling
    plt.ylim(0, df['Avg AoLI'].max() * 1.15)
    
    plt.tight_layout()
    plt.savefig("results/tradeoff_plot.png", dpi=300)
    plt.close() # Closes the figure to free up memory
    print("✅ Grouped Plot saved to results/tradeoff_plot.png")