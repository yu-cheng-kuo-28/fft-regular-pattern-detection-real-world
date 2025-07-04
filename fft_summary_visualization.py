# FFT Analysis Summary and Visualization
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
import pandas as pd
import polars as pl

print("🚀 ENHANCED FFT FREQUENCY DOMAIN ANALYSIS")
print("="*60)

# Load and process the data (replicating the analysis)
df = pl.read_csv("data/2025-06-25.csv", ignore_errors=True)
df_2882 = df.filter(pl.col("product").cast(pl.String) == "2882")
df_filtered = df_2882.filter(
    (pl.col("flag") == 1) & 
    (pl.col("match_time") >= 90000000) & 
    (pl.col("match_time") <= 132500000)
).select(["match_time", "quantity"])

viz_data = df_filtered.to_pandas()
viz_data = viz_data.sort_values('match_time').reset_index(drop=True)
viz_data['time_seconds'] = (viz_data['match_time'] - 90000000) / 1000

# Create 1-second bins
time_end = int(np.ceil(viz_data['time_seconds'].max()))
bin_edges = np.arange(0, time_end + 1, 1)
bin_centers = bin_edges[:-1] + 0.5
volume_binned, _ = np.histogram(viz_data['time_seconds'], bins=bin_edges, weights=viz_data['quantity'])

print(f"📊 Data Summary:")
print(f"   • Total trades: {len(viz_data):,}")
print(f"   • Total bins: {len(volume_binned):,}")
print(f"   • Duration: {time_end/60:.1f} minutes")

# FFT Analysis Parameters
BL = 512  # Block length
Fs = 1    # Sampling frequency
num_blocks = len(volume_binned) // BL

print(f"🔧 FFT Parameters:")
print(f"   • Block length (BL): {BL} samples")
print(f"   • Sampling frequency (Fs): {Fs} Hz")
print(f"   • Available blocks: {num_blocks}")
print(f"   • Frequency resolution: {Fs/BL:.6f} Hz")

# Extract ALL available blocks for comprehensive FFT analysis
blocks = []
for i in range(num_blocks):
    start_idx = i * BL
    end_idx = start_idx + BL
    blocks.append(volume_binned[start_idx:end_idx])

blocks = np.array(blocks)
print(f"   • Blocks extracted: {len(blocks)}")

# Comprehensive FFT Analysis
fft_results = []
psd_results = []
freqs = np.fft.fftfreq(BL, 1/Fs)

# Process each block
for i, block in enumerate(blocks):
    # Remove DC component and apply window
    block_centered = block - np.mean(block)
    window = np.hanning(BL)
    block_windowed = block_centered * window
    
    # Compute FFT
    fft_result = np.fft.fft(block_windowed)
    psd = np.abs(fft_result)**2
    
    fft_results.append(fft_result)
    psd_results.append(psd)

fft_results = np.array(fft_results)
psd_results = np.array(psd_results)

# Average PSD across all blocks for better SNR
avg_psd = np.mean(psd_results, axis=0)

# Focus on positive frequencies only
positive_mask = freqs >= 0
freqs_pos = freqs[positive_mask]
avg_psd_pos = avg_psd[positive_mask]

# Convert frequencies to periods for interpretation
periods_seconds = 1 / freqs_pos[1:]  # Skip DC component
periods_minutes = periods_seconds / 60

print(f"📈 FFT Analysis Complete:")
print(f"   • Frequency bins: {len(freqs_pos)}")
print(f"   • Period range: {periods_seconds.min():.1f}s to {periods_seconds.max():.1f}s")
print(f"   • Period range: {periods_minutes.min():.2f}min to {periods_minutes.max():.1f}min")

# Pattern Detection Analysis
ac_psd = avg_psd_pos[1:]  # Skip DC component
ac_freqs = freqs_pos[1:]
noise_floor = np.median(ac_psd)
peak_threshold = noise_floor * 3

# Find significant peaks
peaks, peak_properties = signal.find_peaks(
    ac_psd, 
    height=peak_threshold,
    distance=5  # Minimum distance between peaks
)

print(f"🎯 Pattern Detection:")
print(f"   • Noise floor: {noise_floor:.2e}")
print(f"   • Peak threshold (3x): {peak_threshold:.2e}")
print(f"   • Significant peaks: {len(peaks)}")

# Create comprehensive visualization with proper frequency domain plots
fig = plt.figure(figsize=(16, 12))
gs = fig.add_gridspec(3, 2, height_ratios=[1, 1, 1], hspace=0.3, wspace=0.3)

# Panel 1: Volume time series (first 3000 seconds)
ax1 = fig.add_subplot(gs[0, :])
time_subset = bin_centers[:min(3000, len(volume_binned))]
volume_subset = volume_binned[:len(time_subset)]
ax1.plot(time_subset, volume_subset, 'orange', linewidth=0.8, alpha=0.7)
ax1.set_xlabel('Time (seconds from 09:00:00)')
ax1.set_ylabel('Volume (shares/second)')
ax1.set_title(f'Volume Time Series (First {len(time_subset)} seconds, {num_blocks} blocks of {BL} samples)')
ax1.grid(True, alpha=0.3)

# Add block boundaries
for i in range(min(6, num_blocks)):
    block_start = i * 512
    if block_start < len(time_subset):
        ax1.axvline(x=block_start, color='red', linestyle='--', alpha=0.6, linewidth=1)
        ax1.text(block_start + 10, max(volume_subset)*0.8, f'Block {i+1}', rotation=90, fontsize=8)

# Panel 2: Power Spectral Density - Linear Scale
ax2 = fig.add_subplot(gs[1, 0])
ax2.plot(freqs_pos[1:], avg_psd_pos[1:], 'darkblue', linewidth=1.5, label='Average PSD')
ax2.axhline(y=noise_floor, color='gray', linestyle=':', alpha=0.8, label='Noise Floor')
ax2.axhline(y=peak_threshold, color='red', linestyle=':', alpha=0.8, label='Detection Threshold (3x)')

# Mark significant peaks if any
if len(peaks) > 0:
    peak_freqs = ac_freqs[peaks]
    peak_powers = ac_psd[peaks]
    ax2.scatter(peak_freqs, peak_powers, color='red', s=50, zorder=5, label=f'{len(peaks)} Significant Peaks')
    
    # Annotate peaks with period information
    for i, (freq, power) in enumerate(zip(peak_freqs, peak_powers)):
        period_sec = 1/freq
        period_min = period_sec/60
        if period_min >= 1:
            ax2.annotate(f'{period_min:.1f}min', 
                        (freq, power), xytext=(5, 5), 
                        textcoords='offset points', fontsize=8)
        else:
            ax2.annotate(f'{period_sec:.0f}s', 
                        (freq, power), xytext=(5, 5), 
                        textcoords='offset points', fontsize=8)

ax2.set_xlabel('Frequency (Hz)')
ax2.set_ylabel('Power Spectral Density')
ax2.set_title('FFT Analysis - Linear Scale')
ax2.legend()
ax2.grid(True, alpha=0.3)
ax2.set_xlim(0, 0.5)  # Show up to Nyquist frequency

# Panel 3: Power Spectral Density - Log Scale
ax3 = fig.add_subplot(gs[1, 1])
ax3.loglog(freqs_pos[1:], avg_psd_pos[1:], 'darkblue', linewidth=1.5, label='Average PSD')
ax3.axhline(y=noise_floor, color='gray', linestyle=':', alpha=0.8, label='Noise Floor')
ax3.axhline(y=peak_threshold, color='red', linestyle=':', alpha=0.8, label='Detection Threshold (3x)')

# Mark significant peaks if any
if len(peaks) > 0:
    ax3.scatter(peak_freqs, peak_powers, color='red', s=50, zorder=5, label=f'{len(peaks)} Significant Peaks')

ax3.set_xlabel('Frequency (Hz)')
ax3.set_ylabel('Power Spectral Density')
ax3.set_title('FFT Analysis - Log-Log Scale')
ax3.legend()
ax3.grid(True, alpha=0.3)

# Panel 4: Individual Block PSDs (first 10 blocks for visualization)
ax4 = fig.add_subplot(gs[2, 0])
colors = plt.cm.viridis(np.linspace(0, 1, min(10, len(blocks))))
for i in range(min(10, len(blocks))):
    ax4.semilogy(freqs_pos[1:], psd_results[i][positive_mask][1:], 
                color=colors[i], alpha=0.6, linewidth=0.8, 
                label=f'Block {i+1}' if i < 5 else "")
ax4.semilogy(freqs_pos[1:], avg_psd_pos[1:], 'red', linewidth=2, label='Average')
ax4.set_xlabel('Frequency (Hz)')
ax4.set_ylabel('Power Spectral Density')
ax4.set_title(f'Individual Block PSDs (showing first {min(10, len(blocks))} blocks)')
ax4.legend()
ax4.grid(True, alpha=0.3)

# Panel 5: Analysis Summary and Results
ax5 = fig.add_subplot(gs[2, 1])
ax5.axis('off')

# Create detailed summary
summary_text = f"""FFT PATTERN DETECTION RESULTS

📊 DATA OVERVIEW:
• Total trades: {len(viz_data):,}
• Analysis duration: {time_end/60:.1f} minutes  
• 1-second bins: {len(volume_binned):,}
• Complete {BL}-sample blocks: {num_blocks}
• Samples analyzed: {num_blocks * BL:,}

🔧 FFT PARAMETERS:
• Sampling rate (Fs): {Fs} Hz
• Block length (BL): {BL} samples
• Frequency resolution: {Fs/BL:.6f} Hz
• Period range: {2/Fs:.0f}-{BL/Fs:.0f} seconds
• Windowing: Hanning window applied

🎯 PATTERN DETECTION:
• Noise floor (median): {noise_floor:.2e}
• Peak threshold (3x noise): {peak_threshold:.2e}
• Significant peaks found: {len(peaks)}
• Max power (excluding DC): {np.max(avg_psd_pos[1:]):.2e}

💡 ANALYSIS METHOD:
• All {num_blocks} blocks processed
• DC component removed per block
• Hanning window applied
• PSD averaged across blocks
• Statistical peak detection used

🎯 CONCLUSION:
{"✅ Regular patterns detected!" if len(peaks) > 0 else "❌ No regular patterns detected"}
{"Market shows algorithmic activity" if len(peaks) > 0 else "Market volume appears random/event-driven"}
"""

ax5.text(0.05, 0.95, summary_text, transform=ax5.transAxes, fontsize=9, 
         verticalalignment='top', fontfamily='monospace',
         bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray", alpha=0.3))

plt.suptitle('Enhanced FFT Frequency Domain Analysis - Stock Volume Pattern Detection', 
             fontsize=14, fontweight='bold')
plt.savefig('enhanced_fft_analysis_summary.png', dpi=300, bbox_inches='tight')
plt.show()

print("🎉 Enhanced FFT Analysis Visualization Created!")
print("📊 Key Improvements:")
print(f"   ✅ Analyzed ALL {num_blocks} available blocks (not just first few)")
print("   ✅ Shows both linear and log-scale frequency domain plots")
print("   ✅ Displays individual block PSDs vs averaged PSD")
print("   ✅ Proper statistical peak detection with threshold")
print("   ✅ Comprehensive analysis summary with all parameters")
print(f"   📈 Frequency resolution: {Fs/BL:.6f} Hz")
print(f"   🕐 Period detection range: {2}s to {BL}s ({BL/60:.1f} minutes)")

if len(peaks) > 0:
    print(f"\n🎯 DETECTED PATTERNS:")
    for i, peak_idx in enumerate(peaks):
        freq = ac_freqs[peak_idx]
        period_sec = 1/freq
        period_min = period_sec/60
        power = ac_psd[peak_idx]
        snr = power / noise_floor
        print(f"   Peak {i+1}: {freq:.6f} Hz, Period: {period_sec:.1f}s ({period_min:.2f}min), SNR: {snr:.1f}x")
else:
    print(f"\n❌ No significant periodic patterns detected in {2}-{BL} second range")
    print("   💡 Consider: different time scales, longer observation, or other assets")
