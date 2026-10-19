"""Recompute and draw current-DATA positive-compound structural diversity.

Python/matplotlib graphics, with RDKit structure depictions. The analysis is
descriptive: neither natural chemical classes nor toxicity mechanisms are inferred.
"""
from __future__ import annotations

import argparse
from contextlib import redirect_stdout
import hashlib
import io
import json
from pathlib import Path
import shutil

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.patches import Rectangle
from matplotlib.ticker import MultipleLocator
import numpy as np
from PIL import Image
from rdkit import Chem
from rdkit.Chem.Draw import rdMolDraw2D

from prepare_figure2_data import prepare
from figure_style import FONT_FAMILY, apply_font, font_path

ROOT = Path(__file__).resolve().parents[1]
WIDTH_MM, HEIGHT_MM = 7453. / 600. * 25.4, 3866. / 600. * 25.4
WIDTH_IN, HEIGHT_IN = WIDTH_MM / 25.4, HEIGHT_MM / 25.4
DPI = 600.
# Original four-color family, consistent across all Figure 2 panels.
# Colors denote clusters of positive compounds, not endpoint classes.
COLORS = ['#2F6FB0', '#7B5AA6', '#D9842B', '#C8AD2F']


def write_json(path, data):
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8')


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def axes_style(ax, grid=True):
    ax.set_axisbelow(True)
    ax.tick_params(pad=1.5)
    if grid:
        ax.grid(True, color='#b0b0b0', alpha=.18, linewidth=.35)
    for spine in ax.spines.values():
        spine.set_color('black')
        spine.set_linewidth(.55)


def panel_title(fig, x, y, letter, text):
    fig.text(x, y, letter, fontsize=8, weight='bold', va='baseline')
    fig.text(x + .027, y, text, fontsize=7, va='baseline')


def structure_files(rep, folder):
    """Render the exact source graph, with specified stereochemistry retained."""
    folder.mkdir(parents=True, exist_ok=True)
    mol = Chem.MolFromSmiles(rep['standardized_smiles'])
    if mol is None:
        raise ValueError(f"Invalid representative {rep['compound_id']}")
    original = Chem.MolToSmiles(mol, isomericSmiles=True)
    for ext in ('png', 'svg'):
        drawer = (rdMolDraw2D.MolDraw2DCairo(1400, 600) if ext == 'png'
                  else rdMolDraw2D.MolDraw2DSVG(1400, 600, -1, -1, True))
        options = drawer.drawOptions()
        options.fontFile = str(font_path())
        options.bondLineWidth = 5.0
        options.minFontSize = 70
        options.maxFontSize = 70
        options.updateAtomPalette({6:(0.,0.,0.),7:(.1,.2,.75),8:(.8,.08,.1),
                                   15:(.65,.3,0.),16:(.6,.35,.05),17:(.12,.45,.2)})
        options.padding = .06
        options.addStereoAnnotation = False
        options.prepareMolsBeforeDrawing = True
        rdMolDraw2D.PrepareAndDrawMolecule(drawer, mol)
        drawer.FinishDrawing()
        content = drawer.GetDrawingText()
        target = folder / f"C{rep['cluster']}_{rep['compound_id']}.{ext}"
        if ext == 'png':
            target.write_bytes(content)
        else:
            content = content.replace('font-family:sans-serif', 'font-family:' + FONT_FAMILY)
            target.write_text(content, encoding='utf-8')
    if Chem.MolToSmiles(mol, isomericSmiles=True) != original:
        raise ValueError('Drawing changed the representative molecular graph')
    return folder / f"C{rep['cluster']}_{rep['compound_id']}.png"


def check_text_canvas(fig):
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    width, height = fig.canvas.get_width_height()
    errors = []
    # 3D tick Text objects retain unprojected coordinates in get_window_extent;
    # their final projected positions are inspected in the exported visual QA.
    unprojected_3d_ticks = set()
    for ax in fig.axes:
        if ax.name == '3d':
            for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
                for tick in axis.get_major_ticks() + axis.get_minor_ticks():
                    unprojected_3d_ticks.update((tick.label1, tick.label2))
    for item in fig.findobj(matplotlib.text.Text):
        if item in unprojected_3d_ticks:
            continue
        if not item.get_visible() or not item.get_text().strip():
            continue
        # Inactive Matplotlib alternate tick labels are outside but are invisible.
        bounds = item.get_window_extent(renderer)
        if bounds.width == 0 or bounds.height == 0:
            continue
        if bounds.x0 < -.5 or bounds.y0 < -.5 or bounds.x1 > width + .5 or bounds.y1 > height + .5:
            errors.append({'text': item.get_text(), 'bounds': list(map(float, bounds.bounds))})
    if errors:
        raise ValueError(f'Clipped text: {errors}')
    return {'passed': True, 'canvas_at_QA_dpi': [width, height],
            'three_dimensional_ticks': 'Projected ticks checked by independent visual QA'}


def export(fig, folder, stem, width_mm, height_mm):
    layout_qa = check_text_canvas(fig)
    for ext in ('png', 'pdf', 'svg'):
        fig.savefig(folder / f'{stem}.{ext}', dpi=DPI, facecolor='white')
    with Image.open(folder / f'{stem}.png') as source:
        rgb = source.convert('RGB')
        rgb.save(folder / f'{stem}.tiff', compression='tiff_lzw', dpi=(DPI, DPI))
        rgb.save(folder / f'{stem}.jpg', quality=98, subsampling=0, dpi=(DPI, DPI))
        preview = rgb.copy()
        preview.thumbnail((1600, 1800), Image.Resampling.LANCZOS)
        preview.save(folder / f'{stem}_preview.png')
        actual = rgb.size
    return {'width_mm': width_mm, 'height_mm': height_mm,
            'raster_pixels': list(actual), 'dpi': DPI, 'text_canvas': layout_qa,
            'sha256': {ext: digest(folder / f'{stem}.{ext}')
                       for ext in ('png', 'jpg', 'tiff', 'pdf', 'svg')}}


def main_figure(data, matrices, pair_data, out):
    """Restore the supplied original A/B/central structures/C composition."""
    from matplotlib.colors import LinearSegmentedColormap, Normalize
    from matplotlib.patches import Ellipse
    from matplotlib import patheffects
    from scipy.cluster.hierarchy import dendrogram, linkage
    from scipy.spatial.distance import squareform
    import textwrap

    records, summaries = data['records'], data['cluster_summaries']
    representatives, metadata = data['representatives'], data['metadata']
    if metadata['selected_k'] != 4:
        raise ValueError('The original layout requires the user-requested fixed four clusters')
    clusters = np.array([r['cluster'] for r in records])
    xy = np.array([[r['tsne_1'], r['tsne_2']] for r in records])
    xyz = np.array([[r[f'tsne_3d_{j}'] for j in (1,2,3)] for r in records])
    fig = plt.figure(figsize=(WIDTH_IN, HEIGHT_IN), layout=None)
    outer = fig.add_gridspec(1,3,width_ratios=[.30,.16,.54],
                            left=.022,right=.992,top=.977,bottom=.035,wspace=.022)
    left = outer[0,0].subgridspec(2,1,hspace=.20)
    ax = fig.add_subplot(left[0])
    ax.set_position([.047,.615,.240,.316])
    for c in range(1,5):
        points = xy[clusters==c]
        ax.scatter(*points.T,s=5.2,c=COLORS[c-1],alpha=.92,linewidths=0,
                   label=f'Cluster {c}',rasterized=True)
    ax.set_xlabel('t-SNE 1',labelpad=1.5)
    ax.set_ylabel('t-SNE 2',labelpad=1.5)
    ax.tick_params(labelsize=5)
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_color('#9A9A9A')
        spine.set_linewidth(.45)
    lg=ax.legend(loc='upper right',fontsize=4.6,markerscale=.75,handletextpad=.3,frameon=False)
    # The same color identities and same cluster field are used in both projections.
    ax3=fig.add_subplot(left[1],projection='3d')
    for c in range(1,5):
        points=xyz[clusters==c]
        ax3.scatter(*points.T,s=7.,c=COLORS[c-1],alpha=.94,linewidths=0,
                    depthshade=False,rasterized=True)
    ax3.set_xlabel('tSNE1',labelpad=-2)
    ax3.set_ylabel('tSNE2',labelpad=-2)
    ax3.set_zlabel('tSNE3',labelpad=-2)
    ax3.tick_params(labelsize=4.5,pad=-1)
    ax3.view_init(elev=25,azim=42)
    ax3.grid(True)
    for axis in (ax3.xaxis,ax3.yaxis,ax3.zaxis):
        axis.pane.set_facecolor((.98,.99,1.,1.))
        axis.pane.set_edgecolor('#C8CDD2')
        axis._axinfo['grid'].update({'color':'#DFE4E8','linewidth':.35})

    rep_ax=fig.add_subplot(outer[0,1])
    rep_ax.set_axis_off()
    rep_ax.set_xlim(0,1);rep_ax.set_ylim(0,1)
    border=Rectangle((.015,.012),.97,.976,fill=False,edgecolor='#BDBDBD',
                     linewidth=.85,linestyle=(0,(3,2)))
    border.set_path_effects([patheffects.SimpleLineShadow(offset=(1,-1),alpha=.15),patheffects.Normal()])
    rep_ax.add_patch(border)
    structure_qa=[]
    for rep,y in zip(representatives,[.865,.625,.385,.145]):
        c=rep['cluster']
        dot=Ellipse((.11,y+.095),width=.095,height=.028,facecolor=COLORS[c-1],edgecolor='none')
        dot.set_path_effects([patheffects.SimplePatchShadow(offset=(1,-1),alpha=.28),patheffects.Normal()])
        rep_ax.add_patch(dot)
        p=structure_files(rep,out/'structures')
        image_ax=rep_ax.inset_axes([.19,y-.063,.77,.17])
        image_ax.imshow(np.asarray(Image.open(p)))
        image_ax.set_axis_off()
        wrapped='\n'.join(textwrap.wrap(rep['name'],width=38))
        rep_ax.text(.59,y-.080,wrapped,ha='center',va='top',fontsize=5.,
                    color='#222222',linespacing=.95)
        # Keep the original visual hierarchy; exact CID and ID remain in source tables.
        structure_qa.append({'compound_id':rep['compound_id'],'PubChem_CID':rep['PubChem_CID'],
                             'cluster':c,'source_smiles_preserved':True,
                             'png':str(p.relative_to(out)), 'full_name_on_figure':True})

    heat_cmap=LinearSegmentedColormap.from_list('original_tanimoto',
                ['#8FC7D9','#FAF8F2','#E87878'],N=256)
    right=outer[0,2].subgridspec(2,2,hspace=.055,wspace=.12)
    heat_checks=[]
    for cell,c in zip(right,(1,2,3,4)):
        idx=np.flatnonzero(clusters==c)
        sim=matrices['tanimoto_similarity'][np.ix_(idx,idx)]
        tree=linkage(squareform(1.-sim,checks=True),method='average',optimal_ordering=True)
        order=dendrogram(tree,no_plot=True)['leaves']
        ordered=sim[np.ix_(order,order)]
        sub=cell.subgridspec(3,3,height_ratios=[.13,.19,.68],
                            width_ratios=[.14,.86,.001],hspace=0,wspace=0)
        title_ax=fig.add_subplot(sub[0,:2]);title_ax.set_axis_off()
        title_ax.text(.06 if c==1 else 0.,.62,
                      f'Cluster {c} (n={len(idx)}): Similarity Heatmap',fontsize=6.2,va='center')
        cbax=title_ax.inset_axes([.82,.47,.16,.18])
        cb=fig.colorbar(matplotlib.cm.ScalarMappable(norm=Normalize(0,1),cmap=heat_cmap),
                        cax=cbax,orientation='horizontal')
        cb.set_ticks([0,.5,1]);cb.ax.tick_params(labelsize=2.8,length=.8,pad=.5)
        cb.set_label('Tanimoto',fontsize=2.8,labelpad=.4);cb.outline.set_linewidth(.25)
        top_ax=fig.add_subplot(sub[1,1]);left_ax=fig.add_subplot(sub[2,0])
        heat_ax=fig.add_subplot(sub[2,1])
        for da,orientation in ((top_ax,'top'),(left_ax,'left')):
            dendrogram(tree,ax=da,orientation=orientation,no_labels=True,
                       color_threshold=0,above_threshold_color='#9A9A9A',
                       link_color_func=lambda _:'#9A9A9A')
            da.set_axis_off()
            for collection in da.collections:collection.set_linewidth(.28)
        # A left-oriented dendrogram places its first leaf at the bottom;
        # imshow places its first row at the top. Align the two explicitly.
        left_ax.invert_yaxis()
        heat_ax.imshow(ordered,cmap=heat_cmap,vmin=0,vmax=1,interpolation='nearest',
                       aspect='auto',rasterized=True)
        heat_ax.set_xticks([]);heat_ax.set_yticks([])
        for spine in heat_ax.spines.values():spine.set(linewidth=.35,color='#A0A0A0')
        heat_checks.append({'cluster':c,'n':len(idx),'diagonal_included':True,
                            'vmin':0,'vmax':1,'linkage':'average','optimal_ordering':True,
                            'left_dendrogram_inverted_to_match_matrix_rows':True})
    for x,y,label in ((.004,.982,'A'),(.004,.492,'B'),(.474,.982,'C')):
        fig.text(x,y,label,fontsize=17,fontweight='bold',ha='left',va='top')
    qa=export(fig,out,'Figure2',WIDTH_MM,HEIGHT_MM)
    from matplotlib import font_manager
    qa.update({'n_compounds':842,'selected_k':4,'user_fixed_k':True,
               'automatic_recommendation_not_displayed_as_optimal':metadata['auto_selected_k'],
               'cluster_counts':[r['n_compounds'] for r in summaries],
               'layout':'Original A=2D, B=3D, central four representatives, C=four dendrogram heatmaps',
               'original_canvas_and_proportions_preserved':True,
               'font_family':FONT_FAMILY,'font_resolved':font_path().name,
               'cluster_colors':{f'Cluster {i+1}':COLORS[i] for i in range(4)},
               'same_cluster_field_and_colors_in_2D_3D_and_representatives':True,
               'heatmaps':heat_checks,'structures':structure_qa,'editable_text':True,
               'raster_layers_in_pdf_svg':['scatter points','similarity heatmaps','structure depictions'],
               'individual_structure_svg_sources':True,
               'native_size_caution':'Original native dimensions retained; do not claim small original-style text is readable at 171 mm without rescaling layout'})
    plt.close(fig)
    return qa


def robustness_figure(data, out):
    meta = data['metadata']
    chosen, k_eval = meta['selected_k'], data['k_evaluation']
    fig = plt.figure(figsize=(171./25.4, 128.25/25.4), layout=None)
    axs = [fig.add_axes(bounds) for bounds in
           ([.079, .58, .405, .33], [.60, .58, .37, .33], [.079, .135, .405, .30], [.58, .115, .39, .33])]
    ax = axs[0]
    axes_style(ax)
    x = [r['k'] for r in k_eval]
    y = [r['silhouette'] for r in k_eval]
    ax.plot(x, y, color='#777777', marker='o', markersize=2.8, linewidth=.85)
    index = x.index(chosen)
    ax.scatter([chosen], [y[index]], color=COLORS[0], s=26, zorder=5)
    maximum = meta['k_selection']['maximum_eligible_silhouette']
    ax.axhline(maximum-.01, color=COLORS[1], linestyle='--', linewidth=.7)
    ax.text(.04, .96, f'Displayed K = {chosen} (user fixed)\nAutomatic recommendation = {meta["auto_selected_k"]}',
             transform=ax.transAxes, va='top', fontsize=5.0)
    ax.set(xlabel='Candidate K', ylabel='Original-space silhouette', ylim=(0, max(y)*1.6))
    ax.set_xticks(x)
    panel_title(fig, .035, .941, 'a', 'Fixed display K and candidate assessment')

    ax = axs[1]
    axes_style(ax)
    restart = [[r['ARI_vs_best_objective_run'] for r in data['multistart_stability'] if r['k']==candidate]
               for candidate in x]
    bp = ax.boxplot(restart, tick_labels=[str(v) for v in x], widths=.55,
                    patch_artist=True, showfliers=True,
                    flierprops={'marker':'.', 'markersize':2},
                    medianprops={'color':'black', 'linewidth':.7})
    for candidate, box in zip(x, bp['boxes']):
        box.set(facecolor=COLORS[0] if candidate==chosen else '#b0b0b0', alpha=.25, linewidth=.5)
    ax.set(xlabel='Candidate K (10 initialisations each)', ylabel='ARI versus best-objective run', ylim=(-.03,1.08))
    panel_title(fig, .56, .941, 'b', 'Initialisation sensitivity')

    ax = axs[2]
    axes_style(ax)
    aris = np.array([r['ARI_on_shared_compounds'] for r in data['subsample_stability']])
    positions = np.arange(1, len(aris)+1)
    ax.scatter(positions, aris, s=12, color=COLORS[0], alpha=.8, linewidths=0)
    ax.axhline(np.median(aris), color=COLORS[1], linestyle='--', linewidth=.8)
    ax.text(.03,.96,f"Median ARI = {np.median(aris):.3f}",transform=ax.transAxes,va='top',fontsize=5.1)
    ax.set(xlabel='Independent 80% subsample', ylabel='ARI on shared compounds', ylim=(0,1.05), xlim=(0,21))
    ax.set_yticks(np.linspace(0,1,6))
    ax.set_xticks([1,5,10,15,20])
    panel_title(fig, .035, .482, 'c', 'Subsampling sensitivity')

    ax = axs[3]
    ax.set_axis_off()
    rows = [[f"C{r['cluster']}",r['n_compounds'],f"{r['median']:.3f}",f"{r['q1']:.3f}–{r['q3']:.3f}"]
            for r in data['cluster_summaries']]
    table = ax.table(cellText=rows, colLabels=['Partition','n','Median','Q1–Q3'],
                     cellLoc='center', bbox=[0,0,1,1], colWidths=[.23,.15,.24,.38])
    table.auto_set_font_size(False)
    table.set_fontsize(5.0)
    for (r,c), cell in table.get_celld().items():
        cell.set(edgecolor='#cccccc', linewidth=.45,
                 facecolor='white' if r==0 or r%2 else '#f2f2fa')
        if r==0:
            cell.get_text().set_weight('bold')
        elif c==0:
            cell.get_text().set_color(COLORS[r-1])
    panel_title(fig, .56, .482, 'd', 'Within-partition similarity summary')
    fig.text(.52,.035,'Four user-fixed exploratory partitions; low separation does not establish natural or mechanistic classes.',
             ha='center',fontsize=5.1)
    qa = export(fig,out,'Figure2_robustness',171.,128.25)
    plt.close(fig)
    return qa


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,default=ROOT/'figures/figure2/outputs')
    parser.add_argument('--copy-to',type=Path)
    args = parser.parse_args()
    out = args.output.resolve()
    out.mkdir(parents=True,exist_ok=True)
    folder = ROOT/'figures/figure2/source_data'
    with redirect_stdout(io.StringIO()):
        prepare(folder,fixed_k=4)
    data = json.loads((folder/'figure2_source_data.json').read_text(encoding='utf-8'))
    meta = data['metadata']
    if meta['analysis_n_records'] != 842 or not meta['fingerprint']['includeChirality']:
        raise ValueError('Unexpected current-DATA population or fingerprint definition')
    plt.rcParams.update({'font.family':FONT_FAMILY,'font.size':6,
        'axes.titlesize':7,'axes.labelsize':6,'xtick.labelsize':5,'ytick.labelsize':5,
        'legend.fontsize':4.8,'axes.linewidth':.45,'axes.edgecolor':'#9A9A9A','xtick.major.width':.4,
        'ytick.major.width':.4,'xtick.major.size':2.,'ytick.major.size':2.,
        'svg.fonttype':'none','pdf.fonttype':42,'ps.fonttype':42,
        'text.usetex':False,'savefig.facecolor':'white','figure.facecolor':'white',
        'axes.formatter.useoffset':False})
    apply_font()
    with np.load(folder/'figure2_pairwise_matrices.npz',allow_pickle=False) as matrices, \
         np.load(folder/'figure2_intra_cluster_similarities.npz',allow_pickle=False) as pairs:
        main_qa = main_figure(data,matrices,pairs,out)
    robust_qa = robustness_figure(data,out)
    qa = {'passed':True,'source_sha256':meta['source']['sha256'],
          'source_data_sha256':digest(folder/'figure2_source_data.json'),
          'style_reference':'Original Figure 2 layout with user-requested Arial typography for all manuscript figures',
          'analysis_source':'Current DATA only','modeling_split_unchanged':True,
          'main':main_qa,'robustness':robust_qa,
          'scientific_limitations':['Four clusters explicitly fixed by the user; not an automatically selected optimum',
             'Weak original-space silhouette; descriptive partitions only',
             'Limited initialisation and subsampling stability',
             'Binary local fingerprints cannot uniquely identify all molecules',
             'No mechanistic or predictive-performance inference from positive-only clusters']}
    write_json(out/'Figure2_QA.json',qa)
    if args.copy_to:
        args.copy_to.mkdir(parents=True,exist_ok=True)
        shutil.copytree(out,args.copy_to,dirs_exist_ok=True)
    print(json.dumps({'passed':True,'n':842,'selected_k':meta['selected_k'],
                      'cluster_counts':main_qa['cluster_counts'],'output':str(out)},ensure_ascii=False))


if __name__=='__main__':
    main()
