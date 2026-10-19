"""Recalculate and draw the seven-descriptor DATA figure in the supplied style."""
from __future__ import annotations

import argparse
from contextlib import redirect_stdout
import hashlib
import io
import json
from pathlib import Path
import shutil
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.patches import Patch
from matplotlib.ticker import FormatStrFormatter, MultipleLocator
import numpy as np
from PIL import Image
from scipy.stats import gaussian_kde

from prepare_figure1_data import prepare
from figure_style import FONT_FAMILY, apply_font, font_path

ROOT = Path(__file__).resolve().parents[1]
ORDER = ['ALogP', 'nAR', 'nHAcc', 'nHDon', 'nRotB', 'TPSA', 'MW']
BLUE, ORANGE = '#1f77b4', '#ff7f0e'
WIDTH_IN = 171.0 / 25.4
HEIGHT_IN = WIDTH_IN * .75
DPI = 4800.0 / WIDTH_IN


def write_json(file, value):
    file.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')


def digest(file):
    return hashlib.sha256(file.read_bytes()).hexdigest()


def p_text(value, stars):
    return ('p < 0.001' if value < .001 else f'p = {value:.3f}') + (' ' + stars)


def add_uniform_legend(ax):
    """Use exactly the same class mapping and legend in all seven panels."""
    handles=[Patch(facecolor=color,alpha=.25,edgecolor='#a0a0a0',linewidth=.3)
             for color in (BLUE,ORANGE)]
    legend=ax.legend(handles,['Positive','Reference-negative'],loc='upper right',
        fontsize=4.8,frameon=True,facecolor='white',edgecolor='#cccccc',framealpha=.9,
        borderpad=.35,labelspacing=.35,handlelength=1.7,handletextpad=.5,
        borderaxespad=.5)
    legend.get_frame().set_linewidth(.45)
    return legend


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=ROOT / 'figures/figure1/outputs')
    parser.add_argument('--copy-to', type=Path, help='Optional second delivery folder for the same final files')
    args = parser.parse_args()
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=True)
    data_folder = ROOT / 'figures/figure1/source_data'
    with redirect_stdout(io.StringIO()):
        prepare(data_folder)
    data = json.loads((data_folder / 'figure1_source_data.json').read_text(encoding='utf-8'))
    records, metadata = data['records'], data['metadata']
    if metadata['label_counts'] != {'0': 831, '1': 842} or len(records) != 1673:
        raise ValueError('Unexpected DATA population')
    labels = np.array([row['label'] for row in records])
    values = {name: np.array([row[name] for row in records], dtype=float) for name in ORDER}
    tests = {row['descriptor']: row for row in data['statistical_tests']}
    spec = json.loads((ROOT / 'figures/figure1/style_spec.json').read_text(encoding='utf-8'))
    plt.rcParams.update({'font.family':FONT_FAMILY, 'font.size':6.0,
        'axes.titlesize':7.0, 'axes.labelsize':6.0, 'xtick.labelsize':5.0,
        'ytick.labelsize':5.0, 'legend.fontsize':4.8, 'axes.linewidth':.55,
        'xtick.major.width':.5, 'ytick.major.width':.5,
        'xtick.major.size':2.5, 'ytick.major.size':2.5,
        'svg.fonttype':'none', 'pdf.fonttype':42, 'ps.fonttype':42,
        'text.usetex':False, 'savefig.facecolor':'white', 'figure.facecolor':'white',
        'axes.formatter.useoffset':False})
    apply_font()
    fig = plt.figure(figsize=(WIDTH_IN, HEIGHT_IN), layout=None)
    plot_checks, geometry, qa_artists, legends = [], [], [], []
    # Preserve equal panel sizes while repairing the inherited left label crop.
    left_px,right_px,gap_ratio=315.,4688.,.272
    panel_width_px=(right_px-left_px)/(3+2*gap_ratio)
    for i, name in enumerate(ORDER):
        x0, y0, x1, y1 = spec['layout']['axes_pixel_boxes_top_origin'][i]['xyxy']
        x0=left_px+(i%3)*panel_width_px*(1+gap_ratio)
        x1=x0+panel_width_px
        box = [x0/4800, 1-y1/3600, (x1-x0)/4800, (y1-y0)/3600]
        ax = fig.add_axes(box)
        ax.set_axisbelow(True)
        ax.tick_params(axis='both',pad=1.5)
        ax.grid(True, color='#b0b0b0', alpha=.18, linewidth=.35)
        for spine in ax.spines.values():
            spine.set_color('black')
            spine.set_linewidth(.55)
        a, b = values[name][labels == 1], values[name][labels == 0]
        ax.set_title('Distribution of ' + name, pad=3.0, weight='normal')
        ax.set_xlabel(name, labelpad=2.0)
        if name in ('ALogP','TPSA','MW'):
            low, high = {'ALogP':(-10.,20.),'TPSA':(0.,550.),'MW':(0.,1050.)}[name]
            if min(a.min(), b.min()) < low:
                low = np.floor(min(a.min(),b.min()))
            if max(a.max(), b.max()) > high:
                high = np.ceil(max(a.max(),b.max()) * 1.03)
            bins = np.linspace(low, high, 31)
            edges, raw_counts = [], []
            for group, color in ((a,BLUE),(b,ORANGE)):
                counts, edge = np.histogram(group, bins=bins)
                if counts.sum() != len(group):
                    raise ValueError(f'Data were lost in the histogram: {name}')
                heights, _, _ = ax.hist(group, bins=bins, density=True, color=color,
                    alpha=.25, edgecolor='#a0a0a0', linewidth=.3)
                if not np.isclose(np.sum(heights * np.diff(edge)), 1.0, atol=1e-12):
                    raise ValueError(f'Histogram density does not integrate to one: {name}')
                edges = edge.tolist()
                raw_counts.append(counts.tolist())
            grid = np.linspace(low, high, 1000)
            for group, color in ((a,BLUE),(b,ORANGE)):
                ax.plot(grid, gaussian_kde(group, bw_method='scott')(grid), color=color, lw=1.0)
            ax.set_xlim(low,high)
            ax.set_ylim(bottom=0)
            ax.set_ylabel('Density', labelpad=1)
            if name == 'ALogP':
                ax.set_xticks(np.arange(-10,21,5))
                ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
            elif name == 'TPSA':
                ax.set_xticks(np.arange(0,551,100))
                ax.yaxis.set_major_formatter(FormatStrFormatter('%.3f'))
            else:
                ax.set_xticks(np.arange(0,1001,200))
                ax.yaxis.set_major_formatter(FormatStrFormatter('%.4f'))
            plot_checks.append({'descriptor':name,'kind':'density','n_positive':len(a),'n_reference_negative':len(b),
                'all_records_in_bins':True,'histogram_class_density_integrals':[1.,1.],
                'bin_edges':edges,'positive_bin_counts':raw_counts[0],'reference_negative_bin_counts':raw_counts[1],
                'xlim':[float(low),float(high)],'KDE_bandwidth':'Scott independently per class; visual smoothing only'})
        else:
            if not np.all(values[name] == np.round(values[name])):
                raise ValueError(f'Count descriptor has noninteger values: {name}')
            low, high = int(values[name].min()), int(values[name].max())
            categories = np.arange(low, high+1)
            max_frequency = 0.
            class_counts = []
            for group, color, offset in ((a,BLUE,-.175),(b,ORANGE,.175)):
                counts = np.array([np.count_nonzero(group == k) for k in categories])
                frequencies = counts / len(group)
                if counts.sum() != len(group) or not np.isclose(frequencies.sum(),1.,atol=1e-12):
                    raise ValueError(f'Data lost from discrete frequency distribution: {name}')
                ax.bar(categories+offset,frequencies,width=.35,color=color,alpha=.25,
                    edgecolor='#a0a0a0',linewidth=.3)
                ax.plot(categories+offset,frequencies,color=color,lw=.9,marker='o',ms=1.8,mew=0)
                max_frequency=max(max_frequency,float(frequencies.max()))
                class_counts.append(counts.tolist())
            if name == 'nAR':
                ticks=categories
            elif name == 'nHDon':
                ticks=categories
            else:
                ticks=np.arange(low,high+1,2)
            ax.set_xticks(ticks)
            ax.set_xlim(low-.65,high+.65)
            ax.set_ylim(0,max_frequency*1.08)
            ax.set_ylabel('Relative frequency',labelpad=2)
            ax.yaxis.set_major_locator(MultipleLocator({'nAR':.05,'nHAcc':.025,'nHDon':.05,'nRotB':.02}[name]))
            ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f' if name in ('nAR','nHDon','nRotB') else '%.3f'))
            plot_checks.append({'descriptor':name,'kind':'relative_frequency','n_positive':len(a),'n_reference_negative':len(b),
                'all_records_shown':True,'class_frequency_sums':[1.,1.],'categories':categories.tolist(),
                'positive_counts':class_counts[0],'reference_negative_counts':class_counts[1],
                'xlim':[low-.65,high+.65]})
        legends.append((name,ax,add_uniform_legend(ax)))
        geometry.append({'descriptor':name,'axes_fraction':box})
        qa_artists.extend([ax.title,ax.xaxis.label,ax.yaxis.label])
        # Locators can return ticks beyond the view; those are not drawn by Axis.
        qa_artists.extend(label for location,label in zip(ax.get_xticks(),ax.get_xticklabels()) if ax.get_xlim()[0] <= location <= ax.get_xlim()[1])
        qa_artists.extend(label for location,label in zip(ax.get_yticks(),ax.get_yticklabels()) if ax.get_ylim()[0] <= location <= ax.get_ylim()[1])
    tx0,ty0,tx1,ty1=spec['layout']['table_pixel_box_top_origin']
    tx0=left_px+panel_width_px*(1+gap_ratio)-10
    table_ax=fig.add_axes([tx0/4800,1-ty1/3600,(tx1-tx0)/4800,(ty1-ty0)/3600])
    table_ax.axis('off')
    contents=[[name,p_text(tests[name]['p_BH_adjusted'],tests[name]['significance_BH'])] for name in ORDER]
    table=table_ax.table(cellText=contents,colLabels=['Molecular Descriptor','Adjusted P value'],
                        cellLoc='center',colLoc='center',loc='center',bbox=[0,0,1,1],colWidths=[.5,.5])
    table.auto_set_font_size(False)
    table.set_fontsize(6.0)
    for (r,c),cell in table.get_celld().items():
        cell.set_edgecolor('#cccccc')
        cell.set_linewidth(.45)
        cell.set_facecolor('white' if r == 0 or r%2 == 1 else '#f2f2fa')
        if r == 0:
            cell.set_text_props(weight='bold')
    # Render with the selected plotting backend and verify all key text fits.
    fig.canvas.draw()
    renderer=fig.canvas.get_renderer()
    bounds=fig.bbox
    legend_checks=[]
    for name,ax,legend in legends:
        lb=legend.get_window_extent(renderer)
        ab=ax.get_window_extent(renderer)
        legend_checks.append({'descriptor':name,'width_px_at_QA_dpi':float(lb.width),
            'height_px_at_QA_dpi':float(lb.height),
            'right_inset_px':float(ab.x1-lb.x1),'top_inset_px':float(ab.y1-lb.y1),
            'labels':[item.get_text() for item in legend.get_texts()],
            'font_sizes_pt':[item.get_fontsize() for item in legend.get_texts()],
            'frame_linewidth_pt':legend.get_frame().get_linewidth()})
    for field in ('width_px_at_QA_dpi','height_px_at_QA_dpi','right_inset_px','top_inset_px'):
        if not np.allclose([item[field] for item in legend_checks],legend_checks[0][field],rtol=0,atol=1e-8):
            raise ValueError('Legends have inconsistent geometry: '+field)
    clipped=[]
    for artist in qa_artists:
        if not artist.get_visible() or not artist.get_text():
            continue
        box=artist.get_window_extent(renderer)
        if box.x0 < bounds.x0-.5 or box.y0 < bounds.y0-.5 or box.x1 > bounds.x1+.5 or box.y1 > bounds.y1+.5:
            clipped.append(artist.get_text())
    if clipped:
        raise ValueError('Text falls outside the figure canvas: '+str(clipped))
    fig.savefig(out/'Figure1.svg',format='svg',bbox_inches=None)
    fig.savefig(out/'Figure1.pdf',format='pdf',bbox_inches=None,metadata={'Title':'Figure 1: molecular descriptor distributions'})
    fig.savefig(out/'Figure1.png',dpi=DPI,bbox_inches=None)
    fig.savefig(out/'Figure1.jpg',dpi=DPI,bbox_inches=None,pil_kwargs={'quality':98,'subsampling':0})
    fig.savefig(out/'Figure1.tiff',dpi=DPI,bbox_inches=None,pil_kwargs={'compression':'tiff_lzw'})
    fig.savefig(out/'Figure1_preview.png',dpi=1600./WIDTH_IN,bbox_inches=None)
    plt.close(fig)
    outputs={}
    for filename in ['Figure1.png','Figure1.jpg','Figure1.tiff','Figure1.pdf','Figure1.svg','Figure1_preview.png']:
        file=out/filename
        outputs[filename]={'sha256':digest(file),'bytes':file.stat().st_size}
    for filename in ['Figure1.png','Figure1.tiff']:
        with Image.open(out/filename) as im:
            if im.size != (4800,3600):
                raise ValueError(f'Unexpected raster dimensions: {filename} {im.size}')
            outputs[filename].update(pixels=list(im.size),dpi=[float(v) for v in im.info.get('dpi',(DPI,DPI))])
    if '<text' not in (out/'Figure1.svg').read_text(encoding='utf-8'):
        raise ValueError('SVG text was unexpectedly outlined')
    qa={'passed':True,'DATA_sha256':metadata['source']['sha256'],'n_records':len(records),
        'label_counts':metadata['label_counts'],'all_seven_descriptors_finite':True,
        'statistics':'Two-sided tie/continuity-corrected asymptotic MWU; BH across seven tests',
        'table_values':contents,'all_key_text_within_canvas':True,'editable_SVG_text':True,
        'width_mm':171.0,'height_mm':128.25,'dpi':DPI,'reference_aspect_ratio_preserved':'4:3',
        'plot_checks':plot_checks,'geometry':geometry,'outputs':outputs,
        'legend_consistency':{'passed':True,'positive_color':BLUE,'reference_negative_color':ORANGE,
            'same_colors_for_bars_and_curves':True,'panels':legend_checks},
        'matplotlib_version':matplotlib.__version__,'font':FONT_FAMILY,
        'font_file_basename':font_path().name,
        'plot_source_policy':'Recomputed from the supplied DATA only; reference raster used solely to specify visual style'}
    write_json(out/'Figure1_QA.json',qa)
    if args.copy_to:
        delivery=args.copy_to.resolve()
        delivery.mkdir(parents=True,exist_ok=True)
        for file in out.iterdir():
            if file.is_file():
                shutil.copy2(file,delivery/file.name)
    print(json.dumps({'output':str(out),'rows':len(records),'raster':[4800,3600],
                      'width_mm':171.,'dpi':DPI,'table':contents,'passed':True},ensure_ascii=False))


if __name__ == '__main__':
    main()
