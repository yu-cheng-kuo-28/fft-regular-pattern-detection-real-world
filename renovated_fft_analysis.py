# Renovated FFT Pattern Detection Analysis
# bin_size=1s, Fs=1Hz, BL=2^9=512 samples

"""
🔬 RENOVATED FFT PATTERN DETECTION: REGULAR BUYING/SELLING PATTERNS
================================================================

SPECIFICATIONS:
• bin_size = 1 second (aggregate volume per second)
• Fs (sampling rate) = 1 Hz
• BL (block length) = 2^9 = 512 samples
• D (duration) = BL/Fs = 512 seconds ≈ 8.5 minutes

OBJECTIVE: Detect regular buying/selling patterns in volume data
"""

import numpy as np
from scipy import fft, signal
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import polars as pl

print("🚀 RENOVATED FFT ANALYSIS: REGULAR BUYING/SELLING PATTERNS")
print("="*60)
print("📊 PARAMETERS:")
print(f"   • bin_size = 1 second")
print(f"   • Fs (sampling rate) = 1 Hz")
print(f"   • BL (block length) = 2^9 = 512 samples")
print(f"   • D (duration) = 512 seconds = {512/60:.1f} minutes")
print(f"   • Frequency resolution = Fs/BL = {1/512:.6f} Hz")
print("="*60)

# Load and prepare the data
print("📁 Loading trading data from CSV...")

# Load the raw data
df = pl.read_csv("data/2025-06-25.csv", ignore_errors=True)
print(f"   • Total records loaded: {len(df):,}")

# Filter for product 2882
df_2882 = df.filter(pl.col("product").cast(pl.String) == "2882")
print(f"   • Product 2882 records: {len(df_2882):,}")

# Apply filters: flag=1 (trade data) and trading hours
df_2882_filtered = df_2882.filter(
    (pl.col("flag") == 1) & 
    (pl.col("match_time") >= 90000000) & 
    (pl.col("match_time") <= 132500000)
).select(["match_time", "quantity"])

print(f"   • Filtered trade records: {len(df_2882_filtered):,}")

# Get the filtered trading data
viz_data = df_2882_filtered.to_pandas()
viz_data = viz_data.sort_values('match_time').reset_index(drop=True)

# Convert match_time to seconds from 09:00:00.000 (90000000)
start_time = 90000000  # 09:00:00.000 in HHMMSSMMM format
viz_data['time_seconds'] = (viz_data['match_time'] - start_time) / 1000  # Convert to seconds

print(f"📊 Original Data:")
print(f"   • Total trades: {len(viz_data):,}")
print(f"   • Time span: {viz_data['time_seconds'].min():.1f}s to {viz_data['time_seconds'].max():.1f}s")
print(f"   • Duration: {(viz_data['time_seconds'].max() - viz_data['time_seconds'].min())/60:.1f} minutes")
print(f"   • Total volume: {viz_data['quantity'].sum():,}")

# Create 1-second bins
time_start = 0  # Start from 09:00:00.000
time_end = int(np.ceil(viz_data['time_seconds'].max()))
bin_edges = np.arange(time_start, time_end + 1, 1)  # 1-second bins
bin_centers = bin_edges[:-1] + 0.5

# Aggregate volume in each 1-second bin
volume_binned, _ = np.histogram(viz_data['time_seconds'], bins=bin_edges, weights=viz_data['quantity'])

print(f"\n📦 Volume Binning Results:")
print(f"   • Number of 1-second bins: {len(volume_binned):,}")
print(f"   • Time range: 0s to {time_end}s ({time_end/60:.1f} minutes)")
print(f"   • Total binned volume: {volume_binned.sum():,}")
print(f"   • Average volume per second: {volume_binned.mean():.1f}")
print(f"   • Max volume in 1-second: {volume_binned.max():,}")
print(f"   • Bins with zero volume: {np.sum(volume_binned == 0):,} ({np.sum(volume_binned == 0)/len(volume_binned)*100:.1f}%)")

# Parameters
BL = 2**9  # Block length = 512 samples
Fs = 1     # Sampling frequency = 1 Hz
D = BL / Fs  # Duration = 512 seconds

print(f"\n🔧 Block Segmentation:")
print(f"   • Block length (BL): {BL} samples")
print(f"   • Sampling frequency (Fs): {Fs} Hz")
print(f"   • Block duration (D): {D} seconds = {D/60:.1f} minutes")
print(f"   • Frequency resolution: {Fs/BL:.6f} Hz")

# Calculate number of complete blocks
num_blocks = len(volume_binned) // BL
total_samples_used = num_blocks * BL

print(f"\n📊 Available Data:")
print(f"   • Total binned samples: {len(volume_binned):,}")
print(f"   • Complete blocks available: {num_blocks}")
print(f"   • Samples used: {total_samples_used:,}")
print(f"   • Samples unused: {len(volume_binned) - total_samples_used}")

if num_blocks == 0:
    print(f"\n⚠️ WARNING: Not enough data for even one complete block!")
    print(f"   Need at least {BL} samples, but only have {len(volume_binned)}")
    print(f"   Consider using shorter block length or more data")
else:
    # Extract blocks for FFT analysis
    blocks = []
    block_times = []
    
    for i in range(num_blocks):
        start_idx = i * BL
        end_idx = start_idx + BL
        block_data = volume_binned[start_idx:end_idx]
        block_time_start = bin_centers[start_idx]
        
        blocks.append(block_data)
        block_times.append(block_time_start)
        
        print(f"   Block {i+1}: samples {start_idx}-{end_idx-1}, time {block_time_start:.0f}s-{block_time_start+D-1:.0f}s")
    
    blocks = np.array(blocks)
    print(f"\n✅ Block extraction complete: {num_blocks} blocks of {BL} samples each")
    
    # FFT parameters
    freq_bins = np.fft.fftfreq(BL, 1/Fs)  # Frequency bins
    freq_resolution = Fs / BL
    
    print(f"\n🔬 FFT Analysis Setup:")
    print(f"   • FFT size: {BL} points")
    print(f"   • Frequency resolution: {freq_resolution:.6f} Hz")
    print(f"   • Frequency range: 0 to {Fs/2:.3f} Hz (Nyquist)")
    print(f"   • Period range: {2/Fs:.1f}s to {BL/Fs:.1f}s")
    
    # Process each block
    fft_results = []
    psd_results = []
    
    for i, block in enumerate(blocks):
        # Remove DC component
        block_centered = block - np.mean(block)
        
        # Apply window to reduce spectral leakage
        window = np.hanning(BL)
        block_windowed = block_centered * window
        
        # Compute FFT
        fft_block = np.fft.fft(block_windowed)
        psd_block = np.abs(fft_block)**2
        
        fft_results.append(fft_block)
        psd_results.append(psd_block)
        
        print(f"   Block {i+1}: Mean volume = {np.mean(block):.1f}, Std = {np.std(block):.1f}")
    
    fft_results = np.array(fft_results)
    psd_results = np.array(psd_results)
    
    # Average PSD across all blocks
    avg_psd = np.mean(psd_results, axis=0)
    
    # Focus on positive frequencies only
    positive_freq_mask = freq_bins >= 0
    freqs_pos = freq_bins[positive_freq_mask]
    avg_psd_pos = avg_psd[positive_freq_mask]
    
    # Convert to periods for interpretation
    periods_seconds = 1 / freqs_pos[1:]  # Skip DC component (freq=0)
    periods_minutes = periods_seconds / 60
    
    print(f"\n📈 FFT Results:")
    print(f"   • Frequency bins: {len(freqs_pos)}")
    print(f"   • DC component power: {avg_psd_pos[0]:.2e}")
    print(f"   • Average AC power: {np.mean(avg_psd_pos[1:]):.2e}")
    print(f"   • Max AC power: {np.max(avg_psd_pos[1:]):.2e}")
    
    # Find significant peaks in the frequency domain
    # Skip DC component (index 0) for peak detection
    ac_psd = avg_psd_pos[1:]
    ac_freqs = freqs_pos[1:]
    ac_periods = periods_seconds
    
    # Set threshold for significant peaks (relative to noise floor)
    noise_floor = np.median(ac_psd)
    peak_threshold = noise_floor * 3  # 3x above median
    
    # Find peaks
    peaks, peak_properties = signal.find_peaks(
        ac_psd, 
        height=peak_threshold,
        distance=5  # Minimum separation between peaks
    )
    
    print(f"\n🎯 PATTERN DETECTION RESULTS:")
    print(f"   • Noise floor (median): {noise_floor:.2e}")
    print(f"   • Peak threshold (3x noise): {peak_threshold:.2e}")
    print(f"   • Significant peaks found: {len(peaks)}")
    
    if len(peaks) > 0:
        # Sort peaks by power (strongest first)
        peak_powers = ac_psd[peaks]
        peak_freqs = ac_freqs[peaks]
        peak_periods = ac_periods[peaks]
        
        sorted_indices = np.argsort(peak_powers)[::-1]
        
        print(f"\n✅ DETECTED REGULAR PATTERNS:")
        for i, idx in enumerate(sorted_indices[:5]):  # Show top 5
            freq = peak_freqs[idx]
            period = peak_periods[idx]
            power = peak_powers[idx]
            snr = power / noise_floor
            
            if period < 60:
                period_str = f"{period:.1f} seconds"
            else:
                period_str = f"{period/60:.1f} minutes"
                
            print(f"   {i+1}. Period: {period_str} (Freq: {freq:.4f} Hz)")
            print(f"       Power: {power:.2e}, SNR: {snr:.1f}x")
            
            # Interpret the pattern
            if 10 <= period <= 30:
                interpretation = "High-frequency algorithmic trading"
            elif 30 <= period <= 120:
                interpretation = "Market making or systematic trading"
            elif 120 <= period <= 300:
                interpretation = "Institutional batch processing"
            else:
                interpretation = "Unknown pattern type"
                
            print(f"       → {interpretation}")
            print()
    else:
        print(f"\n❌ No significant regular patterns detected")
        print(f"   • All frequency components below threshold")
        print(f"   • Market volume appears largely random or irregular")
        print(f"   • Consider different time scales or longer observation periods")
    
    # Create visualization
    print(f"\n🎨 Creating visualization...")
    print(f"   • Time series data points: {len(bin_centers)}")
    print(f"   • Volume range: {volume_binned.min():.1f} to {volume_binned.max():.1f}")
    print(f"   • Frequency bins for PSD: {len(ac_freqs)}")
    print(f"   • PSD range: {ac_psd.min():.2e} to {ac_psd.max():.2e}")
    print(f"   • Noise floor: {noise_floor:.2e}")
    print(f"   • Peak threshold: {peak_threshold:.2e}")
    
    fig = make_subplots(
        rows=3, cols=1,
        subplot_titles=(
            f'Volume Time Series (1-second bins, {len(volume_binned):,} samples)',
            f'Average Power Spectral Density ({num_blocks} blocks of {BL} samples)',
            'Detected Regular Patterns (if any)'
        ),
        vertical_spacing=0.12,
        row_heights=[0.4, 0.4, 0.2]
    )
    
    # Panel 1: Time domain - Volume binned data
    fig.add_trace(
        go.Scatter(
            x=bin_centers[:len(volume_binned)],
            y=volume_binned,
            mode='lines',
            name='Volume (1s bins)',
            line=dict(color='#F18F01', width=1),
            hovertemplate="Time: %{x:.0f}s<br>Volume: %{y}<extra></extra>"
        ),
        row=1, col=1
    )
    
    # Add block boundaries
    for i in range(num_blocks + 1):
        x_pos = i * BL
        if x_pos <= len(volume_binned):
            fig.add_vline(
                x=x_pos, 
                line=dict(color="red", width=1, dash="dash"),
                annotation_text=f"Block {i}" if i < num_blocks else "End",
                annotation_position="top",
                row=1, col=1
            )
    
    # Panel 2: Frequency domain - Average PSD
    fig.add_trace(
        go.Scatter(
            x=freqs_pos[1:],  # Skip DC component
            y=avg_psd_pos[1:],
            mode='lines',
            name='Average PSD',
            line=dict(color='#2E86AB', width=1.5),
            hovertemplate="Frequency: %{x:.4f} Hz<br>Period: %{customdata:.1f}s<br>Power: %{y:.2e}<extra></extra>",
            customdata=periods_seconds
        ),
        row=2, col=1
    )
    
    # Add noise floor and threshold lines
    fig.add_hline(
        y=noise_floor,
        line=dict(color="gray", width=1, dash="dot"),
        annotation_text="Noise Floor",
        annotation_position="right",
        row=2, col=1
    )
    
    fig.add_hline(
        y=peak_threshold,
        line=dict(color="red", width=1, dash="dot"),
        annotation_text="Peak Threshold (3x)",
        annotation_position="right",
        row=2, col=1
    )
    
    # Panel 3: Detected patterns
    if len(peaks) > 0:
        fig.add_trace(
            go.Scatter(
                x=peak_freqs,
                y=peak_powers,
                mode='markers+text',
                name='Detected Patterns',
                marker=dict(color='red', size=8, symbol='circle'),
                text=[f'{p:.0f}s' if p < 60 else f'{p/60:.1f}m' for p in peak_periods],
                textposition='top center',
                hovertemplate="Frequency: %{x:.4f} Hz<br>Period: %{text}<br>Power: %{y:.2e}<extra></extra>"
            ),
            row=3, col=1
        )
    else:
        # Show message when no patterns detected
        fig.add_annotation(
            text="No significant regular patterns detected",
            xref="x3", yref="y3",
            x=0.25, y=0.5,
            showarrow=False,
            font=dict(size=14, color="gray"),
            row=3, col=1
        )
    
    # Update layout
    fig.update_layout(
        title=f'Renovated FFT Analysis: Regular Buying/Selling Pattern Detection<br>'
              f'Fs=1Hz, BL={BL}, D={D/60:.1f}min, {num_blocks} blocks analyzed',
        height=800,
        showlegend=True,
        plot_bgcolor='white',
        paper_bgcolor='white'
    )
    
    # Update axes
    fig.update_xaxes(title_text='Time (seconds)', showgrid=True, gridcolor='lightgray', row=1, col=1)
    fig.update_xaxes(title_text='Frequency (Hz)', type='log', showgrid=True, gridcolor='lightgray', row=2, col=1)
    fig.update_xaxes(title_text='Frequency (Hz)', type='log', showgrid=True, gridcolor='lightgray', row=3, col=1)
    
    fig.update_yaxes(title_text='Volume (shares/second)', showgrid=True, gridcolor='lightgray', row=1, col=1)
    fig.update_yaxes(title_text='Power Spectral Density', type='log', showgrid=True, gridcolor='lightgray', row=2, col=1)
    fig.update_yaxes(title_text='Pattern Power', type='log', showgrid=True, gridcolor='lightgray', row=3, col=1)
    
    # Save the figure to HTML file for viewing
    output_file = "renovated_fft_analysis_results.html"
    fig.write_html(output_file)
    print(f"\n📊 Visualization saved to: {output_file}")
    
    # Also try to display if in interactive environment
    try:
        fig.show()
    except Exception as e:
        print(f"   Note: Interactive display not available ({e})")
    
    print(f"\n📊 Visualization complete!")
    
    # Final Summary
    print(f"\n🎯 RENOVATED FFT ANALYSIS SUMMARY")
    print(f"="*60)
    
    print(f"📊 ANALYSIS PARAMETERS:")
    print(f"   • Bin size: 1 second")
    print(f"   • Sampling rate (Fs): {Fs} Hz")
    print(f"   • Block length (BL): {BL} samples")
    print(f"   • Block duration (D): {D} seconds = {D/60:.1f} minutes")
    print(f"   • Frequency resolution: {freq_resolution:.6f} Hz")
    
    print(f"\n📈 DATA SUMMARY:")
    print(f"   • Original trades: {len(viz_data):,}")
    print(f"   • 1-second bins: {len(volume_binned):,}")
    print(f"   • Complete blocks: {num_blocks}")
    print(f"   • Total volume analyzed: {volume_binned[:total_samples_used].sum():,} shares")
    
    print(f"\n🔍 PATTERN DETECTION RESULTS:")
    if len(peaks) > 0:
        print(f"   ✅ Significant patterns found: {len(peaks)}")
        print(f"   📊 Strongest pattern period: {periods_seconds[peaks[sorted_indices[0]]]:.1f} seconds")
        print(f"   ⚡ Best signal-to-noise ratio: {peak_powers[sorted_indices[0]]/noise_floor:.1f}x")
        
        # Trading implications
        print(f"\n💡 TRADING IMPLICATIONS:")
        strongest_period = periods_seconds[peaks[sorted_indices[0]]]
        if 10 <= strongest_period <= 30:
            print(f"   🚨 High-frequency algorithmic activity detected")
            print(f"   → Consider HFT competition and latency optimization")
        elif 30 <= strongest_period <= 120:
            print(f"   📈 Systematic trading patterns detected")
            print(f"   → Potential market making or regular rebalancing")
        elif 120 <= strongest_period <= 300:
            print(f"   🏢 Institutional batch processing detected")
            print(f"   → Large orders being executed systematically")
        
        print(f"\n🎯 RECOMMENDED ACTIONS:")
        print(f"   1. Monitor these regular patterns for trading opportunities")
        print(f"   2. Adjust execution timing to avoid/exploit pattern phases")
        print(f"   3. Consider pattern-based market making strategies")
        print(f"   4. Watch for pattern disruptions (market regime changes)")
        
    else:
        print(f"   ❌ No significant regular patterns detected")
        print(f"   📊 Market volume appears largely random")
        
        print(f"\n💡 IMPLICATIONS:")
        print(f"   • Low algorithmic trading activity in this timeframe")
        print(f"   • Volume driven by irregular market events")
        print(f"   • Consider longer observation periods or different assets")
        print(f"   • Focus on other trading signals (price action, volatility)")
    
    print(f"\n" + "="*60)
    print(f"🎉 RENOVATED FFT ANALYSIS COMPLETE!")
    print(f"   🔧 Method: 1s binning + {BL}-sample FFT blocks")
    print(f"   🎯 Focus: Regular buying/selling pattern detection")
    print(f"   📈 Result: {'Patterns detected' if len(peaks) > 0 else 'No regular patterns found'}")
    print(f"   ⚡ Status: Ready for trading strategy development")
    print(f"="*60)
