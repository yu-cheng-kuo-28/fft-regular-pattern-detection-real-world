# Static FFT Visualization using Matplotlib
# Alternative visualization for the renovated FFT analysis

import numpy as np
from scipy import fft, signal
import matplotlib.pyplot as plt
import pandas as pd
import polars as pl

print("🎨 STATIC FFT VISUALIZATION (Matplotlib)")
print("="*50)

# Re-run the core analysis (simplified version)
def run_fft_analysis():
    # Load and prepare the data
    df = pl.read_csv("data/2025-06-25.csv", ignore_errors=True)
    df_2882 = df.filter(pl.col("product").cast(pl.String) == "2882")
    df_2882_filtered = df_2882.filter(
        (pl.col("flag") == 1) & 
        (pl.col("match_time") >= 90000000) & 
        (pl.col("match_time") <= 132500000)
    ).select(["match_time", "quantity"])
    
    viz_data = df_2882_filtered.to_pandas()
    
    # Parameters
    Fs = 1  # Hz
    BL = 512  # Block length
    D = BL / Fs  # Duration per block
    
    # Create time series
    time_range = np.arange(0, viz_data['match_time'].max() + 1)
    volume_binned = np.zeros(len(time_range))
    
    for _, row in viz_data.iterrows():
        time_idx = int(row['match_time'])
        if time_idx < len(volume_binned):
            volume_binned[time_idx] += row['quantity']
    
    bin_centers = time_range
    
    # Block processing
    num_blocks = len(volume_binned) // BL
    blocks = []
    for i in range(num_blocks):
        start_idx = i * BL
        end_idx = start_idx + BL
        block_data = volume_binned[start_idx:end_idx]
        blocks.append(block_data)
    
    blocks = np.array(blocks)
    
    # FFT analysis
    freq_bins = np.fft.fftfreq(BL, 1/Fs)
    psd_results = []
    
    for block in blocks:
        block_centered = block - np.mean(block)
        window = np.hanning(BL)
        block_windowed = block_centered * window
        
        fft_result = np.fft.fft(block_windowed)
        psd = np.abs(fft_result) ** 2
        psd_results.append(psd)
    
    psd_results = np.array(psd_results)
    avg_psd = np.mean(psd_results, axis=0)
    
    # Extract positive frequencies only
    positive_indices = freq_bins > 0
    ac_freqs = freq_bins[positive_indices]
    ac_psd = avg_psd[positive_indices]
    ac_periods = 1.0 / ac_freqs
    
    # Pattern detection
    noise_floor = np.median(ac_psd)
    peak_threshold = 3 * noise_floor
    
    # Find peaks
    peak_indices, _ = signal.find_peaks(ac_psd, height=peak_threshold)
    
    return {
        'bin_centers': bin_centers,
        'volume_binned': volume_binned,
        'ac_freqs': ac_freqs,
        'ac_psd': ac_psd,
        'ac_periods': ac_periods,
        'noise_floor': noise_floor,
        'peak_threshold': peak_threshold,
        'peak_indices': peak_indices,
        'num_blocks': num_blocks,
        'BL': BL
    }

# Run analysis
data = run_fft_analysis()

# Create static visualization
fig, axes = plt.subplots(3, 1, figsize=(12, 10))
fig.suptitle('Renovated FFT Analysis: Regular Buying/Selling Pattern Detection\n' + 
             f'Fs=1Hz, BL={data["BL"]}, {data["num_blocks"]} blocks analyzed', 
             fontsize=14, fontweight='bold')

# Panel 1: Time series
axes[0].plot(data['bin_centers'][:5000], data['volume_binned'][:5000], 
             color='blue', alpha=0.7, linewidth=0.5)
axes[0].set_title(f'Volume Time Series (1-second bins, first 5000 samples shown)')
axes[0].set_xlabel('Time (seconds)')
axes[0].set_ylabel('Volume (shares/second)')
axes[0].grid(True, alpha=0.3)

# Panel 2: Power Spectral Density
axes[1].loglog(data['ac_freqs'], data['ac_psd'], color='blue', alpha=0.7, linewidth=1)
axes[1].axhline(y=data['noise_floor'], color='orange', linestyle='--', 
                label=f'Noise Floor ({data["noise_floor"]:.2e})')
axes[1].axhline(y=data['peak_threshold'], color='red', linestyle=':', 
                label=f'Peak Threshold (3x) ({data["peak_threshold"]:.2e})')
axes[1].set_title(f'Average Power Spectral Density ({data["num_blocks"]} blocks)')
axes[1].set_xlabel('Frequency (Hz)')
axes[1].set_ylabel('Power Spectral Density')
axes[1].legend()
axes[1].grid(True, alpha=0.3)

# Panel 3: Detected patterns
if len(data['peak_indices']) > 0:
    peak_freqs = data['ac_freqs'][data['peak_indices']]
    peak_powers = data['ac_psd'][data['peak_indices']]
    peak_periods = data['ac_periods'][data['peak_indices']]
    
    axes[2].semilogx(peak_freqs, peak_powers, 'ro', markersize=8, label='Detected Patterns')
    
    # Add period labels
    for i, (freq, power, period) in enumerate(zip(peak_freqs, peak_powers, peak_periods)):
        if period < 60:
            label = f'{period:.1f}s'
        else:
            label = f'{period/60:.1f}m'
        axes[2].annotate(label, (freq, power), xytext=(5, 5), 
                        textcoords='offset points', fontsize=8)
    
    axes[2].set_title(f'Detected Regular Patterns ({len(data["peak_indices"])} found)')
    axes[2].legend()
else:
    axes[2].text(0.5, 0.5, 'No significant regular patterns detected', 
                ha='center', va='center', transform=axes[2].transAxes,
                fontsize=12, color='gray')
    axes[2].set_title('Detected Regular Patterns (None Found)')

axes[2].set_xlabel('Frequency (Hz)')
axes[2].set_ylabel('Pattern Power')
axes[2].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('static_fft_analysis_results.png', dpi=300, bbox_inches='tight')
plt.savefig('static_fft_analysis_results.pdf', bbox_inches='tight')

print(f"✅ Static visualization saved to:")
print(f"   • static_fft_analysis_results.png")
print(f"   • static_fft_analysis_results.pdf")

# Summary
print(f"\n📊 VISUALIZATION SUMMARY:")
print(f"   • Time series points plotted: {min(5000, len(data['volume_binned']))}")
print(f"   • Frequency bins: {len(data['ac_freqs'])}")
print(f"   • PSD range: {data['ac_psd'].min():.2e} to {data['ac_psd'].max():.2e}")
print(f"   • Patterns detected: {len(data['peak_indices'])}")

if len(data['peak_indices']) > 0:
    peak_periods = data['ac_periods'][data['peak_indices']]
    print(f"   • Pattern periods: {', '.join([f'{p:.1f}s' if p < 60 else f'{p/60:.1f}m' for p in peak_periods])}")
else:
    print(f"   • No regular patterns above threshold")

print(f"\n🎯 RESULT: {'Patterns detected' if len(data['peak_indices']) > 0 else 'No regular patterns found'}")
