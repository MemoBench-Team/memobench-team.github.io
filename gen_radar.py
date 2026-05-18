"""Generate radar charts for MemoBench website — style matching reference."""
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe

# ── Data ──────────────────────────────────────────────────────────────────────
models = [
    'LingBot-World', 'Wan2.2', 'FantasyWorld', 'Matrix-Game 2.0',
    'SVC', 'Open-SoRA', 'LTX-Video', 'CogVideoX',
]

# Automated metrics (7 axes)
auto_labels = ['VisQual', 'MotSmooth', 'ObjConsist', '3DConsist',
               'ORS×100', 'CamCtrl', 'ImgReward']
auto_data = {
    'LingBot-World':   [47.4, 57.6, 59.0, 88.2, 38.1, 37.4, 36.7],
    'Wan2.2':          [40.0, 54.0, 50.7, 84.5, 32.8, 29.8, 26.1],
    'FantasyWorld':    [51.0, 55.2, 47.6, 88.7, 27.6, 27.2, 30.7],
    'Matrix-Game 2.0': [61.2, 83.6, 46.5, 93.7, 15.7, 17.3, 22.3],
    'SVC':             [43.3, 63.1, 59.5, 88.5, 29.4, 65.2, 22.3],
    'Open-SoRA':       [49.7, 68.3, 47.2, 89.7, 18.2, 16.8, 31.3],
    'LTX-Video':       [44.9, 84.4, 81.6, 94.1, 33.0, 17.1, 37.1],
    'CogVideoX':       [40.1, 59.8, 54.0, 94.0, 25.1, 12.0, 34.9],
}

# VQA dimensions (4 axes)
vqa_labels = ['Inst.Following', 'Obj.&Bkg.', 'Cont.Memory', 'Phys.Adh.']
vqa_data = {
    'LingBot-World':   [64.2, 44.4, 42.1, 53.6],
    'Wan2.2':          [50.6, 30.2, 36.8, 38.9],
    'FantasyWorld':    [50.5, 25.6, 37.1, 33.6],
    'Matrix-Game 2.0': [37.5, 12.7, 36.5, 21.8],
    'SVC':             [49.7, 23.8, 29.6, 33.3],
    'Open-SoRA':       [43.2, 66.8, 48.3, 59.7],
    'LTX-Video':       [41.0, 76.6, 57.0, 63.5],
    'CogVideoX':       [40.5, 52.4, 42.7, 42.8],
}

# Pastel-ish, saturated colors matching the reference style
colors = [
    '#4A90D9',  # LingBot - blue
    '#9B59B6',  # Wan - purple
    '#E91E8C',  # Fantasy - pink
    '#2ECC71',  # Matrix - green
    '#E67E22',  # SVC - orange
    '#E74C3C',  # Open-SoRA - coral/red
    '#F39C12',  # LTX - golden
    '#95A5A6',  # CogVideo - gray
]


def make_radar(labels, data_dict, title, out_path, figsize=(9, 9)):
    N = len(labels)
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=figsize, subplot_kw=dict(polar=True))
    fig.patch.set_facecolor('white')

    # Light blue background on radar area only
    ax.set_facecolor((0.86, 0.92, 0.98, 0.5))

    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)

    # ── Scaling ───────────────────────────────────────────────────────────
    all_vals = [v for vals in data_dict.values() for v in vals]
    max_val = max(all_vals)
    r_max = int(np.ceil(max_val / 20) * 20)

    ax.set_ylim(0, r_max)
    r_ticks = list(range(20, r_max + 1, 20))
    ax.set_yticks(r_ticks)
    ax.set_yticklabels([])

    # Remove default x tick labels
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels([])

    # ── Grid styling ──────────────────────────────────────────────────────
    ax.spines['polar'].set_visible(False)
    ax.grid(color='#c0c8d0', linewidth=0.8, alpha=0.6)

    # Score numbers along the top axis (angle=0, i.e. 12 o'clock)
    for rt in r_ticks:
        ax.text(0, rt + 1.5, str(int(rt)),
                ha='center', va='bottom',
                fontsize=10, color='#5a6a7a', fontweight='500')

    # ── Axis labels outside the circle ────────────────────────────────────
    for i, (angle, label) in enumerate(zip(angles[:-1], labels)):
        angle_deg = np.degrees(angle) % 360
        # Place labels just outside the outer ring
        label_r = r_max * 1.12
        if angle_deg < 5 or angle_deg > 355:
            ha, va = 'center', 'bottom'
        elif angle_deg < 90:
            ha, va = 'left', 'bottom'
        elif 85 < angle_deg < 95:
            ha, va = 'left', 'center'
        elif angle_deg < 180:
            ha, va = 'left', 'top'
        elif 175 < angle_deg < 185:
            ha, va = 'center', 'top'
        elif angle_deg < 270:
            ha, va = 'right', 'top'
        elif 265 < angle_deg < 275:
            ha, va = 'right', 'center'
        else:
            ha, va = 'right', 'bottom'

        ax.text(angle, label_r, label,
                ha=ha, va=va,
                fontsize=14, fontweight='bold', color='#2c3e50')

    # ── Plot each model ───────────────────────────────────────────────────
    for idx, model in enumerate(models):
        vals = data_dict[model]
        vals_closed = vals + vals[:1]
        ax.plot(angles, vals_closed, '-o',
                color=colors[idx], linewidth=2.5, markersize=6,
                markerfacecolor='white', markeredgewidth=2,
                markeredgecolor=colors[idx], label=model, alpha=0.9,
                zorder=3 + idx)
        ax.fill(angles, vals_closed, color=colors[idx], alpha=0.08)

    # ── Legend at the bottom, horizontal ──────────────────────────────────
    legend = ax.legend(
        loc='upper center',
        bbox_to_anchor=(0.5, -0.08),
        ncol=4,
        fontsize=11,
        frameon=False,
        handletextpad=0.5,
        columnspacing=1.5,
        markerscale=1.3,
    )
    # Make legend markers larger filled circles
    for handle in legend.legend_handles:
        handle.set_markersize(10)
        handle.set_markerfacecolor(handle.get_color())
        handle.set_markeredgecolor(handle.get_color())
        handle.set_linewidth(0)

    # ── Title ─────────────────────────────────────────────────────────────
    title_pad = 45 if N <= 4 else 42
    ax.set_title(title, fontsize=18, fontweight='bold', pad=title_pad, color='#1a1a2e')

    plt.tight_layout()
    fig.savefig(out_path, dpi=180, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.close(fig)
    print(f"Saved: {out_path}")


if __name__ == '__main__':
    make_radar(auto_labels, auto_data,
               'Model Performance Overview',
               '/overlay/website/images/radar_chart.png',
               figsize=(9, 9))

    make_radar(vqa_labels, vqa_data,
               'VQA Performance Overview',
               '/overlay/website/images/radar_vqa.png',
               figsize=(9, 9))
