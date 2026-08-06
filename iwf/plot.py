import numpy as np
from pyscf.data.nist import HARTREE2EV, BOHR

from plotly import graph_objects as go
from plotly.subplots import make_subplots
from plotly import io as pio


def _serve_on_localhost(fig, port):
    import webbrowser
    from http.server import BaseHTTPRequestHandler, HTTPServer

    html = pio.to_html(fig, full_html=True, include_plotlyjs=True).encode('utf8')

    class OneShotRequestHandler(BaseHTTPRequestHandler):
        def do_GET(self):
            self.send_response(200)
            self.send_header('Content-type', 'text/html')
            self.end_headers()
            self.wfile.write(html)

        def log_message(self, format, *args):
            pass

    server = HTTPServer(('127.0.0.1', port), OneShotRequestHandler)
    webbrowser.open(f'http://localhost:{port}')
    server.handle_request()


def _special_kpt_idxs(special_kpts, path, kpts_rel):
    idxs, cur = [], 0
    for lbl in path:
        ref = special_kpts[lbl]
        for i in range(cur, len(kpts_rel)):
            if np.allclose(kpts_rel[i], ref, atol=1e-6):
                idxs.append(i)
                cur = i + 1
                break
    return idxs


class BandPlotter():
    def __init__(self,
                 ase_obj, kpath: str,
                 kpts_rel=None, kpts_abs=None, npoints=None,
                 special_points=None,
                 energy_range=[-10, 10], # eV
                 nrow=1, ncol=1,
                 ):
        """"""
        self.is_subplot = (nrow > 1 or ncol > 1)

        if npoints is not None:
            bp = ase_obj.cell.bandpath(kpath, npoints=npoints)
            special_points = bp.special_points
            kpts_rel = bp.kpts

        if kpts_rel is not None:
            recip = ase_obj.cell.reciprocal().array
            kpts_abs = kpts_rel @ (recip * 2 * np.pi * BOHR)

        kpt_idxs = _special_kpt_idxs(special_points, kpath, kpts_rel)
        kpt_dists = np.linalg.norm(np.diff(kpts_abs, axis=0), axis=1)
        kpt_x = np.concatenate([[0.0], np.cumsum(kpt_dists)])
        ktick_vals = [float(kpt_x[int(idx)]) for idx in kpt_idxs]

        self._legend_for = {}
        if self.is_subplot:
            self.fig = make_subplots(rows=nrow, cols=ncol, horizontal_spacing=0.03)
            for r in range(1, nrow + 1):
                for c in range(1, ncol + 1):
                    n = (r - 1) * ncol + c
                    suffix = '' if n == 1 else str(n)
                    legend_name = 'legend' if n == 1 else f'legend{n}'
                    self._legend_for[(r, c)] = legend_name
                    xdom = self.fig.layout[f'xaxis{suffix}'].domain
                    ydom = self.fig.layout[f'yaxis{suffix}'].domain
                    x_inset = 0.08 * (xdom[1] - xdom[0])
                    self.fig.update_layout(**{legend_name: dict(
                        x=xdom[1] - x_inset, y=ydom[1], xanchor='right', yanchor='top',
                        bgcolor='rgba(255,255,255,1)',
                        bordercolor='lightgrey', borderwidth=1,
                        font=dict(size=16),
                    )})
        else:
            self.fig = go.Figure()

        self.kpt_x = kpt_x
        self.ktick_vals = ktick_vals

        self.fig.update_xaxes(tickmode='array', tickvals=ktick_vals, ticktext=list(kpath),
                     showgrid=True, gridcolor='black', gridwidth=1)
        self.fig.update_yaxes(range=energy_range)

        if self.is_subplot:
            self.fig.update_xaxes(title='k-path', row=nrow)
            self.fig.update_yaxes(title='Energy (eV)', col=1)
            for c in range(2, ncol + 1):
                self.fig.update_yaxes(showticklabels=False, col=c)
        else:
            self.fig.update_xaxes(title='k-path')
            self.fig.update_yaxes(title='Energy (eV)')


    _LEGEND_POSITIONS = {
        'top right': dict(x=1.0, y=1.0, xanchor='right', yanchor='top'),
        'top left': dict(x=0.0, y=1.0, xanchor='left', yanchor='top'),
        'bottom right': dict(x=1.0, y=0.0, xanchor='right', yanchor='bottom'),
        'bottom left': dict(x=0.0, y=0.0, xanchor='left', yanchor='bottom'),
    }

    def add_abinitio_band(self, mo_energy=None, group_name=None, anal_data=None,
                          unit='au', mode='orig', nrow=None, ncol=None,
                          name=None, color=None, symbol=None, char_size=16,
                          show_char_text=False, char_text_size=None, legend_pos=None,
                          show_legend=True):

        if mode == 'orig':
            if nrow is None or ncol is None:
                show_legend_global = True
            else:
                show_legend_global = (nrow == 1 and ncol == 1)

            mo_e_eV = mo_energy * HARTREE2EV if unit == 'au' else mo_energy
            traces = [
                go.Scatter(
                    x=self.kpt_x, y=ew, mode='lines',
                    name='original', showlegend=(i == 0) and show_legend_global and show_legend,
                    line=dict(color='#bbbbbb', width=2, dash='dot')
                    ) for i, ew in enumerate(mo_e_eV.T)
                ]
        elif mode == 'lo':
            if name is None: name = 'IAO'
            if color is None: color = '#1565c0'
            if symbol is None: symbol = 'circle'
            mo_e_eV = mo_energy * HARTREE2EV if unit == 'au' else mo_energy
            traces = [
                go.Scatter(
                    x=self.kpt_x, y=ew, mode='markers+lines',
                    name=name, showlegend=(i == 0) and show_legend,
                    line=dict(color=color, width=2, dash='dot'),
                    marker=dict(color=color, size=6, symbol=symbol)
                    ) for i, ew in enumerate(mo_e_eV.T)
                ]
        elif mode == 'char':
            traces = []
            grp_color = ["#cd9644","#48C4B5", "#9858C0", "#4a86ca"]
            for g, (grp_name, grp_data) in enumerate(zip(group_name, anal_data)):
                print([len(grp_data_k['char']) for grp_data_k in grp_data])
                legend_name = "char: " + grp_name
                for i, grp_data_k in enumerate(grp_data):
                    ew, char, occ = grp_data_k['ew'], grp_data_k['char'], grp_data_k['occ']
                    scatter_kwargs = dict(
                        x=[self.kpt_x[i]] * len(ew),
                        y=ew * HARTREE2EV if unit == 'au' else ew,
                        mode='markers+text' if show_char_text else 'markers',
                        name=legend_name,
                        legendgroup=legend_name,
                        showlegend=False,
                        marker=dict(
                            color=grp_color[g],
                            size=char_size,
                            opacity=np.clip(np.clip(char, 0.0, 1.0), 0.00, 1.0),
                            symbol='circle',
                        ),
                        customdata=np.stack([char, occ], axis=-1),
                        hovertemplate=(
                            grp_name + '<br>'
                            'E = %{y:.3f} eV<br>'
                            'char = %{customdata[0]:.2f}<br>'
                            'occ = %{customdata[1]:.2f}'
                            '<extra></extra>'
                        ),
                    )
                    if show_char_text:
                        text_size = char_text_size if char_text_size is not None else round(char_size * 0.625)
                        scatter_kwargs['text'] = [f'{c:.2f}' for c in char]
                        scatter_kwargs['textposition'] = 'top center'
                        scatter_kwargs['textfont'] = dict(size=text_size, color=grp_color[g])
                    traces.append(go.Scatter(**scatter_kwargs))
                # dummy trace with a fixed marker size, so the legend swatch
                # doesn't inherit a tiny size from whichever k-point is first
                traces.append(go.Scatter(
                    x=[None], y=[None], mode='markers',
                    name=legend_name, legendgroup=legend_name,
                    showlegend=show_legend,
                    marker=dict(color=grp_color[g], size=char_size, symbol='circle'),
                ))
        else:
            raise ValueError(f"Unknown mode: {mode}")

        if (nrow, ncol) == (None, None):
            assert not self.is_subplot, "Cannot add band to subplot without specifying row/col"
            self.fig.add_traces(traces)
            legend_name = 'legend'
        else:
            legend_name = self._legend_for.get((nrow, ncol), 'legend')
            for tr in traces:
                tr.update(legend=legend_name)
            self.fig.add_traces(traces, rows=[nrow]*len(traces), cols=[ncol]*len(traces))

        if legend_pos is not None:
            if isinstance(legend_pos, str):
                legend_dict = self._LEGEND_POSITIONS.get(legend_pos)
                if legend_dict is None:
                    raise ValueError(
                        f"Unknown legend_pos '{legend_pos}'. "
                        f"Choose from {list(self._LEGEND_POSITIONS)} or pass a dict."
                    )
            else:
                legend_dict = legend_pos
            self.fig.update_layout(**{legend_name: legend_dict})

    def add_window(self, dis_win=None, dis_froz=None, nrow=None, ncol=None,
                   color=None, froz_color=None):
        """
        Draw Wannier90 disentanglement window boundaries on a (sub)plot.

        dis_win    : (min, max) outer disentanglement window in eV, or None
        dis_froz   : (min, max) inner (frozen) window in eV, or None
        color      : line color for the outer window (default '#333333');
                     pass the band's color to match the two
        froz_color : line color for the inner window (default '#d95f02');
                     falls back to `color` if not given
        """
        if color is None: color = '#333333'
        if froz_color is None: froz_color = color if color is not None else '#d95f02'
        loc = dict(row=nrow, col=ncol) if self.is_subplot else {}
        if dis_win is not None:
            win_min, win_max = dis_win
            line = dict(color=color, width=1.5, dash='dash')
            self.fig.add_hline(y=win_min, line=line, **loc)
            self.fig.add_hline(y=win_max, line=line, annotation_text='outer window',
                               annotation_position='top left',
                               annotation_font=dict(size=11, color=color), **loc)
        if dis_froz is not None:
            froz_min, froz_max = dis_froz
            line = dict(color=froz_color, width=1.5, dash='dot')
            self.fig.add_hline(y=froz_min, line=line, **loc)
            self.fig.add_hline(y=froz_max, line=line, annotation_text='inner window',
                               annotation_position='top left',
                               annotation_font=dict(size=11, color=froz_color), **loc)

    def dump(self, target=8050):
        self.fig.update_layout(
            font=dict(family='Times New Roman', size=28),
            template='plotly_white',
        )

        if isinstance(target, int): # port
            _serve_on_localhost(self.fig, target)

        elif '.png' in target:
            pio.base_renderers.default = 'png'
            pio.write_image(self.fig, target, scale=2, width=1200, height=800)
            print(f'Band plot written to {target}')
