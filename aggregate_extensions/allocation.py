import sys
sys.path.append('c:\\s\\telos\\python\\aggregate_project')

import itertools
import logging
import aggregate as agg
import matplotlib.pyplot as plt
import pandas as pd
import matplotlib
import numpy as np
import re

# fontsize : int or float or {'xx-small', 'x-small', 'small', 'medium', 'large', 'x-large', 'xx-large'}
matplotlib.rcParams['legend.fontsize'] = 'xx-small'

aggdevlog  = logging.getLogger('aggdev.log')
aggdevlog.setLevel(logging.INFO)


class Fairness():
    """
    try various marginal combos to see how benefit is allocated
    """

    def __init__(self, agg_spec_dict, bs, log2, subports='all'):
        """
        agg_spec_dict is a dictionary label : agg_clause. The agg clause must include
        the initial \\t and the ending \\n.
        """
        self.agg_spec_dict = agg_spec_dict
        self.uw = agg.Underwriter(update=False, create_all=False)
        self.ports = {}
        agg_names = list(self.agg_spec_dict.keys())
        if subports=='all':
            subports = [''.join(i) for i in self.subsets()]
        self.loss = None
        for k in subports:
            aggdevlog.info(f'Creating portfolio {k}')
            pgm = ''
            for l in agg_names:
                if l in k:
                    pgm += agg_spec_dict[l]
            self.ports[k] = self.uw(f'port {k}\n{pgm}\n')
            self.ports[k].update(bs=bs, log2=log2, add_exa=True, padding=2, trim_df=True, remove_fuzz=True)
            if self.loss is None:
                self.loss = self.ports[k].density_df.loss
        # create other variables: anss = answer_s stores answers from exhibit factory; base is
        # the base calibration class; uber is a consolidation of ans.exhibit output from the factory
        self.anss = None
        self.base = None
        self.uber = None
        self.last_plot = None

    def __getitem__(self, item):
        return self.ports[item]

    @property
    def last_figure(self):
        """
        last fig or list of figs
        for convenience it always returns a list
        :return:
        """
        if type(self.last_plot) == list:
            ans = []
            for x in self.last_plot:
                try:
                    if len(x):
                        ans.append(x.flatten()[0].get_figure())
                except TypeError:
                    ans.append(x.get_figure())
            return ans
        else:
            try:
                return [self.last_plot.get_figure()]
            except:
                return self.last_plot.flatten()[0].get_figure()
    # @property
    # def stand_alone_mapping(self):
    #     """
    #     maps a line name to its corresponding stand-alone portfolio
    #     :return:
    #     """

    def items(self):
        """
        for n, v in self.items()

        :return:
        """
        return self.ports.items()

    def subsets(self):
        """
        all non empty subportfolios
        """
        x = self.agg_spec_dict.keys()
        return list(itertools.chain.from_iterable(
            itertools.combinations(x,n) for n in range(len(x) + 1)))[1:]

    def make_vars(self, p, kind):
        '''
        return dictionary var_dict is a dictionary line->var at the given threshold
        also maps subportfolio to var
        :param p:
        :param kind:
        :return:
        '''
        ans = {n if len(v.line_names) > 1 else v.line_names[0] : v.q(p, kind) for n, v in self.ports.items()}
        # append the single line portfolios by portfolio name
        for n, v in self.ports.items():
            if len(v.line_names) == 1:
                ans[n] = v.q(p, kind)
        return ans

    def plot_density(self, example='all', nc=3, **kwargs):
        """
        plot the densities
        for time being go with 3 cols
        """
        if example == 'all':
            what_2_plot = self.ports.keys()
        else:
            what_2_plot = example
        ax = SimpleAxes(len(what_2_plot), nc=nc)
        for nm in what_2_plot:
            port = self.ports[nm]
            port.density_df.filter(regex='p_').plot(logy=True, ax=next(ax), **kwargs)
        ax.tidy()
        self.last_plot = ax

    def gamma(self, p, kind, port_name, plot=False, compute_stand_alone=False, loss_threshold=-1,
              ylim_zoom=(.98, 1.005), extreme_var=1-2e-8):
        """
        return the vector gamma_a(x), the conditional layer effectiveness given assets a
        gamma can be created with no base and no calibration. It only depends on total losses.
        It does NOT vary by line - because of equal priority.
        a must be in index?

        All sublines of port_name must appear in fair as stand alone portfolios

        """
        port = self.ports[port_name]
        temp = port.density_df.filter(regex='^p_|^e1xi_1gta_|exi_xgta_|exi_xeqa_|exeqa_|S|loss').copy()
        var_dict = self.make_vars(p, kind)
        extreme_var_dict = self.make_vars(extreme_var, 'upper')
        # temp = port.density_df.filter(regex='^p_|^e1xi_1gta_|S|loss').copy()
        # just before...
        a = var_dict[port_name]
        a_ = a - port.bs

        # total is a special case
        l = 'total'

        # ?? makes the revised calc below match this calc with offsets
        # add_exa has some cumsum vs. cumintegral issues...
        # the alt calc is much closer to raw output, does not divide and mult by S(x) and does
        # not have the offset issues, so we are using it.
        # old
        # xinv = temp[f'e1xi_1gta_{l}'].shift(-1)
        # gam_name = f'gamma_{port_name}_{l}'
        # s = temp.S
        # temp[gam_name] = 0
        # temp.loc[0:a_, gam_name] = (s[0:a_] - s[a] + a * xinv[a]) / s[0:a_]
        # temp.loc[a:, gam_name] = a * xinv[a:] / s[a:]

        # all the other lines, gamma_{a, i}
        # rate of payment factor
        min_xa = np.minimum(temp.loss, a) / temp.loss
        temp['min_xa_x'] = min_xa

        # for total
        gam_name = f'gamma_{port_name}_{l}'
        # unconditional version avoid divide and multiply by a small number
        # exeqa is closest to the raw output...
        # there is an INDEX ISSUE with _add_exa...there are cumsums there that should be cumintegrals..
        temp[f'exi_x1gta_{l}'] = np.cumsum((temp[f'exeqa_{l}'] * temp.p_total / temp.loss)[::-1]) * port.bs
        #                               this V 1.0 is exi_x for total
        temp[gam_name] = np.cumsum((min_xa * 1.0 * temp.p_total)[::-1]) / \
                         (temp[f'exi_x1gta_{l}']) * port.bs

        if len(port.line_names) > 0:
            # if only one line then it is the same as total
            # although oddly you seem to get a different answer...?
            for l in port.line_names:
                # old version just computes the factor using the line probs but the higher capital
                # this information can be also obtained by running gamma() on the single line portfolio
                # sa = stand alone; need to figure the sa capital
                if compute_stand_alone:
                    a_l = var_dict[l]
                    a_l_ = a_l - port.bs
                    xinv = temp[f'e1xi_1gta_{l}'].shift(-1)
                    gam_name = f'gamma_{l}_sa'
                    s = 1-temp[f'p_{l}'].cumsum()
                    temp[f'S_{l}'] = s
                    temp[gam_name] = 0
                    temp.loc[0:a_l_, gam_name] = (s[0:a_l_] - s[a_l] + a_l * xinv[a_l]) / s[0:a_l_]
                    temp.loc[a_l:, gam_name] = a_l * xinv[a_l:] / s[a_l:]
                    temp[gam_name] = temp[gam_name].shift(1)
                    # method unreliable in extreme right tail
                    temp.loc[extreme_var_dict[l]:, gam_name] = np.nan

                # revised version
                gam_name = f'gamma_{port_name}_{l}'
                # unconditional version avoid divide and multiply by a small number
                # exeqa is closest to the raw output...
                # there is an INDEX ISSUE with _add_exa...there are cumsums there that should be cumintegrals..
                temp[f'exi_x1gta_{l}'] = np.cumsum((temp[f'exeqa_{l}'] * temp.p_total / temp.loss )[::-1]) * port.bs
                temp[gam_name] = np.cumsum((min_xa * temp[f'exi_xeqa_{l}'] * temp.p_total)[::-1]) / \
                                 (temp[f'exi_x1gta_{l}']) * port.bs

        # other stuff that is easy to ignore but helpful for plotting
        temp['BEST'] = 1
        temp.loc[a:, 'BEST'] = a / temp.loc[a:, 'loss']
        temp['WORST'] = np.maximum(0, 1 - temp.loc[a, 'S'] / temp.S)

        if plot:
            if loss_threshold > 0:
                renamer = {
                    "gamma_A.thin_sa": f"A stand alone, a={var_dict['A.thin']:.0f}",
                    "gamma_AB_A.thin": "A part of A+B",
                    "gamma_AB_B.thick": "B part of A+B",
                    "gamma_AB_total": f"A+B, a={var_dict['AB']:.0f}",
                    "gamma_B.thick_sa": f"B stand alone, a={var_dict['B.thick']:.0f}",
                    "p_A.thin": "A density",
                    "p_B.thick": "B density",
                    "p_total": "A+B density",
                    "S_A.thin": "A survival",
                    "S_B.thick": "B survival",
                    "S_total": "A+B survival"
                }

            # rename if doing the second set of plots too
            if loss_threshold > 0:
                spl = temp.filter(regex='gamma|BEST|WORST').rename(columns=renamer).sort_index(1).\
                    plot(ylim=[-.05, 1.05], linewidth=1)
            else:
                spl = temp.filter(regex='gamma|BEST|WORST').sort_index(1).plot(ylim=[-.05, 1.05], linewidth=1)
            for ln in spl.lines:
                lbl = ln.get_label()
                if lbl == lbl.upper():
                    ln.set(linewidth=1, linestyle='--', label=lbl.title())
            spl.grid('b')
            if loss_threshold > 0:
                spl.set(xlim=[0, loss_threshold])
            # update to reflect changes to line styles
            spl.legend()
            self.last_plot = spl

            # fancy comparison plots
            if loss_threshold > 0 and compute_stand_alone:
                temp_ex = temp.query(f'loss < {loss_threshold}').filter(regex='gamma_|p_')
                temp_ex[[f'S_{l[2:]}' for l in temp_ex.filter(regex='p_').columns]] = (1 - temp_ex.filter(regex='p_').cumsum())

                f, axs = plt.subplots(3, 1, figsize=(8, 9), squeeze=False, constrained_layout=True)
                ax1 = axs[0, 0]
                ax2 = axs[1, 0]
                ax3 = axs[2, 0]

                temp_ex.filter(regex='gamma_').rename(columns=renamer).sort_index(axis=1).plot(ax=ax1)
                temp_ex.filter(regex='gamma_').rename(columns=renamer).sort_index(axis=1).plot(ax=ax2)
                ax2.legend(loc='lower left')
                ax1.set(ylim=ylim_zoom, title='Gamma Functions (Zoom)')
                ax2.set(ylim=[0, 1.005], title='Gamma Functions')
                ax3.set(ylim=[0, 1.005], title=f'Survival Functions up to p={port.cdf(loss_threshold):.1%} Loss')
                temp_ex.filter(regex='S_').rename(columns=renamer).sort_index(axis=1).plot(ax=ax3)
                # sort out colors
                col_by_line = {l.get_label().split()[0]: l.get_color() for l in ax3.lines}
                l_loc = dict(zip(axs.flatten(), ['upper right', 'lower left', 'upper right']))
                for ax in axs.flatten():
                    for line in ax.lines:
                        # line name and shortened line name
                        ln = line.get_label()
                        lns = ln.split()[0].strip(',')
                        ls = '--' if ln.find('stand') > 0 else '-'
                        line.set(color=col_by_line[lns], linestyle=ls)
                    # update legend to reflect line style changes
                    ax.legend(loc=l_loc[ax])
                for ax in axs.flatten():
                    ax.grid()
                    # ax.set(xlim=[0, var['AB']])
                self.last_plot = [spl, axs.flatten()]
        else:
            self.last_plot = None
        return temp.sort_index(axis=1)

    def gammas(self, p, kind, base_name, subportfolios='all', plot=False, compute_stand_alone=False):
        """
        make summary over selected subportfolios or all

        :param p: percentile for assets
        :param kind: upper or lower var
        :base_name: the name of the portfolio containing all the lines
        :return:
        """
        temp = pd.DataFrame(index=self.loss)
        if subportfolios == 'all':
            subportfolios = self.ports.items()
        for n, port in subportfolios:
            if n != base_name:
                g = self.gamma(p, kind, n, plot, compute_stand_alone).\
                    filter(regex='gamma_[\d\D]*_(?!total)')
            else:
                # all columns
                g = self.gamma(p, kind, n, plot, compute_stand_alone).filter(regex='gamma_')
            temp[g.columns] = g
        port = self[base_name]
        # pull in the conditional expectations
        for ln in port.line_names_ex:
            temp[f'exeqa_{ln}'] = port.density_df[f'exeqa_{ln}']
            temp[f'exgta_{ln}'] = port.density_df[f'exgta_{ln}']
            temp[f'exlea_{ln}'] = port.density_df[f'exlea_{ln}']
        return temp

    def calibrate(self, dname, LR=None, ROE=None, p=None, A=None, base_port=-1, plot=False):
        # calibrate the base distortion that will be used for all the other examples
        if type(base_port) == int:
            base_port_name = list(self.ports.keys())[base_port]
        else:
            base_port_name = base_port
        # base is an Answer class object with lots of info
        self.base = self.ports[base_port_name].\
            example_factory(dname=dname, LR=LR, ROE=ROE, p=p, A=A, index='loss', plot=plot)
        if plot:
            # also plot the distortion function
            ax = SimpleAxes(1, 1)
            self.base.distortion.plot(ax=next(ax))
            self.last_plot = ax
        # show the highlights in return
        return self.base.exhibit.T.filter(regex='^(M\.|T\.|EP|A)', axis=0)


    def rerun(self, p, plot=False):
        """
        re run for sub portfolios at possibly different capital level using
        the base distortion
        E.g. can see how a line in the base compares with the same line stand alone
        or in other portfolios. Or can see how changing the capital requirement
        alters the ROEs. Generally, if you increase p then you lower the ROE because
        you use more lower cost high layer capital.
        """
        self.anss = {}
        for port in self.ports.values():
            self.anss[port.name] = port.example_factory(dname=self.base.distortion,
                                                        p=p, index='loss', plot=plot)

        self.uber = pd.concat([i.exhibit.T.filter(regex='^(M\.|T\.|EP|A)', axis=0).sort_index() for i in self.anss.values()],
                 axis=1, keys=list(self.anss.keys()), names=['Example'], sort=True)

        return self.uber

    def plot_example(self, example, plot=True, xlim=0):
        """
        plausible plots for the specified example and a summary table

        The line names must be of the form [A-Z]anything and identified by the capital letter

        self = fair
        example = 'AB'
        xlim = 0
        plot = True


        """
        junk = self.anss[example].augmented_df
        port = self.ports[example]
        temp = junk.filter(regex='exi_xgtag?_(?!sum)|^S|^gS').copy()

        # rename columns
        def rename(cols):
            ans = {}
            for c in cols:
                c1 = re.sub('exi_xgta_([A-Z]).*', r'alpha.\1', c)
                if c1 != c:
                    ans[c] = c1
                    continue
                c1 = re.sub('exi_xgtag_([A-Z]).*', r'beta.\1', c)
                if c1 != c:
                    ans[c] = c1
                    continue
                c1 = re.sub('exi_xeqa_([A-Z]).*', r'E[\1 | X=a]/a', c)
                if c1 != c:
                    ans[c] = c1
                    continue
            return ans

        renamer = {'S': 'M.L', 'gS': 'M.P'}
        renamer.update(rename(junk.columns))
        temp = temp.rename(columns=renamer)
        # add extra stats
        temp['M.LR'] = temp['M.L'] / temp['M.P']
        temp['M.ROE'] = (temp['M.P'] - temp['M.L']) / (1 - temp['M.P'])
        temp['M.M'] = temp['M.P'] - temp['M.L']
        for l in port.line_names:
            # irritation that you can't have one letter line names
            l0 = l[0]
            temp[f'beta/alpha.{l0}'] = temp[f'beta.{l0}'] / temp[f'alpha.{l0}']
            temp[f'M.M.{l0}'] = temp[f'beta.{l0}'] * temp['M.P'] - temp[f'alpha.{l0}'] * temp['M.L']

        if plot:
            def tidy(simple_ax):
                ax = simple_ax.ax
                ax.tick_params('x', which='major', labelsize='small')
                ax.tick_params('y', which='major', labelsize='x-small')
                ax.title.set_size('medium')

            junk.index.name = 'Assets a'
            temp.index.name = 'Assets a'
            if xlim == 0:
                xlim = port.q(1 - 1e-5)
            ax = SimpleAxes(8, sharex=True)

            (1 - junk.filter(regex='p_').cumsum()).rename(columns=renamer).sort_index(1). \
                plot(ylim=[0, 1], xlim=[0, xlim], title='Survival functions', ax=next(ax)).grid('b')
            tidy(ax)

            junk.filter(regex='exi_xgtag?').rename(columns=renamer).sort_index(1). \
                plot(ylim=[0, 1], xlim=[0, xlim], title=r'$\alpha=E[X_i/X | X>a],\beta=E_Q$ by Line', ax=next(ax)).grid(
                'b')
            tidy(ax)

            # total margins
            junk.filter(regex='^T\.M').rename(columns=renamer).sort_index(1). \
                plot(xlim=[0, xlim], title='Total Margins by Line', ax=next(ax)).grid('b')
            tidy(ax)

            # marginal margins
            (junk.filter(regex='^M\.M').rename(columns=renamer).sort_index(1) / port.bs). \
                plot(ylim=[-.1, .4], xlim=[0, xlim], title='Marginal Margins by Line', ax=next(ax)).grid('b')
            tidy(ax)

            l = junk.filter(regex='^Q|gF').rename(columns=renamer).sort_index(1). \
                plot(xlim=[0, xlim], title='Capital = 1-gS = F!', ax=next(ax))
            l.grid('b')
            l.lines[0].set(linewidth=5, alpha=0.36)
            tidy(ax)

            # see apply distortion, line 1890 ROE is in augmented_df
            junk.filter(regex='^ROE$|exi_xeqa').rename(columns=renamer).sort_index(1). \
                plot(xlim=[0, xlim], title='M.ROE Total and $E[X_i/X | X=a]$ by line', ax=next(ax)).grid('b')
            tidy(ax)

            # improve scale selection
            temp.filter(regex='beta/alpha\.|LR').rename(columns=renamer).sort_index(1). \
                plot(ylim=[-.05, 1.5], xlim=[0, xlim], title='Alpha, Beta and Marginal LR',
                     ax=next(ax)).grid('b')
            tidy(ax)

            junk.filter(regex='LR').rename(columns=renamer).sort_index(1). \
                plot(ylim=[-.05, 1.25], xlim=[0, xlim], title='Total ↑LR by Line',
                     ax=next(ax)).grid('b')
            tidy(ax)

            ax.tidy()
            # store away
            self.last_plot = ax.axs
        return temp


# general utilities
class SimpleAxes():
    def __init__(self, n, nc=3, aspect=1.5, sm_height=2.0, lg_height=4, **kwargs):
        """
        make a reasonable grid of n axes nc per row with given height and aspect ratio
        returns the figure, the axs and ax iterator
        sm_height uses for two or more rows
        lg_height used for just one row

        kwargs passed to subplots, e.g. sharex sharey
        """
        nc = min(nc, n)
        nr = n // nc
        if n % nc:
            nr += 1
        if nr == 1:
            height = sm_height
        else:
            height = lg_height
        w = nc * height * aspect
        h = nr * height
        sc = min(w, 8) / w
        w *= sc
        h *= sc
        w = min(w, 8)
        self.f, self.axs = plt.subplots(nr, nc, figsize=(w, h), squeeze=False,
                              constrained_layout=True, **kwargs)
        self.axit = iter(self.axs.flatten())
        self._ax = None

    def __next__(self):
        self._ax = next(self.axit)
        return self._ax

    def get_figure(self):
        return self.f

    def tidy(self):
        """
        remove all unused plots
        """
        try:
            while 1:
                self.f.delaxes(next(self.axit))
        except StopIteration:
            return

    @property
    def ax(self):
        if self._ax is None:
            self._ax = next(self.axit)
        return self._ax

