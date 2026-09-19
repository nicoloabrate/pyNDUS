from dataclasses import dataclass, field
from typing import Optional, Tuple


@dataclass(frozen=True, order=True)
class SensitivityChannel:
    """
    ENDF-aware identifier for one sensitivity-profile axis entry.

    ``average_MF``/``average_MT`` identify the mean (best-estimate) nuclear-data quantity 
    to which the sensitivity refers. 
    ``covariance_MF``/``covariance_MT`` identify
    the primary covariance channel used by sandwich calculations. 
    Additional valid covariance MTs, such as MF34/MT2 for the first elastic Legendre
    moment, can be stored in ``covariance_MT_aliases``. ``L`` is used for
    angular-distribution Legendre moments.
    """

    average_MF: Optional[int]
    average_MT: Optional[int]
    covariance_MF: Optional[int] = None
    covariance_MT: Optional[int] = None
    L: Optional[int] = None
    name: Optional[str] = field(default=None, compare=False)
    covariance_MT_aliases: Tuple[int, ...] = field(default=(),
                                                  compare=False)

    @property
    def MT(self):
        """Return the average ENDF MT number."""
        return self.average_MT

    @classmethod
    def from_alias(cls, alias):
        """Build a channel from a known human/readers' label."""
        label = " ".join(str(alias).split()).lower()
        if label in _ALIASES:
            return _ALIASES[label]
        tokens = label.split()
        if tokens in (["total", "xs"], ["xs", "total"]):
            mt = 1
        elif len(tokens) == 3 and tokens[0] == "mt" and tokens[2] == "xs":
            mt = int(tokens[1])
        elif len(tokens) == 3 and tokens[0] == "xs" and tokens[1] == "mt":
            mt = int(tokens[2])
        elif len(tokens) == 2 and tokens[0] == "xs":
            mt = int(tokens[1])
        else:
            mt = None
        if mt is not None:
            return cls.from_endf(average_MF=3, average_MT=mt)
        raise ValueError(f"Unknown sensitivity channel alias {alias!r}.")

    @classmethod
    def from_endf(cls, *, average_MF=None, average_MT=None, covariance_MF=None, 
                  covariance_MT=None, L=None, name=None):
        """
        Build a channel from average-side or covariance-side ENDF identifiers.

        If the supplied identifiers match a registered ENDF quantity, the
        registered channel is returned. Otherwise a generic channel is created
        when enough information is available.
        """
        has_average_pair = average_MF is not None and average_MT is not None
        has_covariance_pair = (
            covariance_MF is not None and covariance_MT is not None)
        if has_average_pair or has_covariance_pair:
            matches = []
            for channel in set(_ALIASES.values()):
                if average_MF is not None and channel.average_MF != average_MF:
                    continue
                if average_MT is not None and channel.average_MT != average_MT:
                    continue
                if covariance_MF is not None and channel.covariance_MF != covariance_MF:
                    continue
                if covariance_MT is not None and covariance_MT not in channel.covariance_MTs:
                    continue
                if L is not None and channel.L != L:
                    continue
                matches.append(channel)

            if len(matches) == 1:
                return matches[0]
            if len(matches) > 1:
                raise ValueError(
                    "ENDF identifiers match multiple sensitivity channels: "
                    f"{sorted(matches)}. Provide L or a more specific channel."
                )

        if average_MF is None and covariance_MF == 33:
            average_MF = 3
        if average_MT is None and covariance_MT is not None:
            average_MT = covariance_MT
        if covariance_MF is None and average_MF is not None:
            covariance_MF = _COVARIANCE_MF_BY_AVERAGE_MF.get(average_MF)
        if covariance_MT is None and average_MT is not None:
            covariance_MT = average_MT
        if name is None and average_MT is not None:
            name = f"MT {average_MT}"

        return cls( average_MF=average_MF, average_MT=average_MT, 
                    covariance_MF=covariance_MF, covariance_MT=covariance_MT, 
                    L=L, name=name, )

    def matches_covariance(self, MF, MT):
        """Return whether this channel can be paired with covariance MF/MT."""
        return self.covariance_MF == int(MF) and int(MT) in self.covariance_MTs

    @property
    def covariance_MTs(self):
        """Return the primary covariance MT plus accepted aliases."""
        if self.covariance_MT is None:
            return ()
        return (self.covariance_MT, ) + tuple(self.covariance_MT_aliases)


_COVARIANCE_MF_BY_AVERAGE_MF = {
                                    1: 31,
                                    3: 33,
                                    4: 34,
                                    5: 35,
                                }

_MCNP_XS_LABELS_TO_MT = {
    "total": 1,
    "elastic": 2,
    "inelastic": 4,
    "n,2nd": 11,
    "n,2n": 16,
    "n,3n": 17,
    "fission": 18,
    "n,f": 19,
    "first-chance fission": 19,
    "(first-chance fission)": 19,
    "n,2nf": 20,
    "second-chance fission": 20,
    "(second-chance fission)": 20,
    "n,nalpha": 22,
    "n,n3alpha": 23,
    "n,2nalpha": 24,
    "n,np": 28,
    "n,n2alpha": 29,
    "n,2n2alpha": 30,
    "n,nd": 32,
    "n,nt": 33,
    "n,n3he": 34,
    "n,nd2alpha": 35,
    "n,nt2alpha": 36,
    "n,4n": 37,
    "n,3nf": 38,
    "fourth-chance fission": 38,
    "(fourth-chance fission)": 38,
    "n,2np": 41,
    "n,3np": 42,
    "n,n2p": 44,
    "n,npalpha": 45,
    "n,gamma": 102,
    "n,p": 103,
    "n,d": 104,
    "n,t": 105,
    "n,3he": 106,
    "n,alpha": 107,
}


NUBAR_TOTAL = SensitivityChannel(1, 452, 31, 452, name="nubar total")
NUBAR_PROMPT = SensitivityChannel(1, 456, 31, 456, name="nubar prompt")
NUBAR_DELAYED = SensitivityChannel(1, 455, 31, 455, name="nubar delayed")
CHI_TOTAL = SensitivityChannel(None, None, None, None, name="chi total")
CHI_PROMPT = SensitivityChannel(5, 18, 35, 18, name="chi prompt")
CHI_DELAYED = SensitivityChannel(5, 455, 35, 455, name="chi delayed")
ELASTIC_LEGENDRE_P1 = SensitivityChannel(4, 2, 34, 251, L=1, name="elastic Legendre moment 1", covariance_MT_aliases=(2, ))
ELASTIC_LEGENDRE_P2 = SensitivityChannel(4, 2, 34, 2, L=2, name="elastic Legendre moment 2")
SCATTER_LEGENDRE_P1 = SensitivityChannel(4, 1, 34, 1, L=1, name="scatter Legendre moment 1")
SCATTER_LEGENDRE_P2 = SensitivityChannel(4, 1, 34, 1, L=2, name="scatter Legendre moment 2")
INELASTIC_LEGENDRE_P1 = SensitivityChannel(4, 4, 34, 4, L=1, name="inelastic Legendre moment 1")
INELASTIC_LEGENDRE_P2 = SensitivityChannel(4, 4, 34, 4, L=2, name="inelastic Legendre moment 2")
ELASTIC_XS = SensitivityChannel(3, 2, 33, 2, name="elastic")
INELASTIC_XS = SensitivityChannel(3, 4, 33, 4, name="inelastic")
FISSION_XS = SensitivityChannel(3, 18, 33, 18, name="fission")
CAPTURE_XS = SensitivityChannel(3, 102, 33, 102, name="capture")
N_XN_XS = SensitivityChannel(3, 16, 33, 16, name="n,xn")
N_ALPHA_XS = SensitivityChannel(3, 107, 33, 107, name="n,alpha")

_REGISTERED_XS_CHANNELS_BY_MT = {
    2: ELASTIC_XS,
    4: INELASTIC_XS,
    16: N_XN_XS,
    18: FISSION_XS,
    102: CAPTURE_XS,
    107: N_ALPHA_XS,
}


_SERPENT_XS_LABELS_TO_MT = {
    "total xs": 1,
    "xs total": 1,
    "mt 2 xs": 2,
    "xs mt 2": 2,
    "xs 2": 2,
    "mt 4 xs": 4,
    "xs mt 4": 4,
    "xs 4": 4,
    "mt 16 xs": 16,
    "xs mt 16": 16,
    "xs 16": 16,
    "mt 18 xs": 18,
    "xs mt 18": 18,
    "xs 18": 18,
    "mt 102 xs": 102,
    "xs mt 102": 102,
    "xs 102": 102,
    "mt 107 xs": 107,
    "xs mt 107": 107,
    "xs 107": 107,
}


def _xs_channel(label, mt):
    return _REGISTERED_XS_CHANNELS_BY_MT.get(
        mt, SensitivityChannel(3, mt, 33, mt, name=label))


_ALIASES = {
            label: _xs_channel(label, mt)
            for label, mt in _MCNP_XS_LABELS_TO_MT.items()
            }

_ALIASES.update({
                 label: _xs_channel(label, mt)
                 for label, mt in _SERPENT_XS_LABELS_TO_MT.items()
                 })

_ALIASES.update({
    "nubar total": NUBAR_TOTAL,
    "total nu": NUBAR_TOTAL,
    "total nubar": NUBAR_TOTAL,
    "nubar prompt": NUBAR_PROMPT,
    "prompt nubar": NUBAR_PROMPT,
    "nubar delayed": NUBAR_DELAYED,
    "delayed nubar": NUBAR_DELAYED,
    "prompt nu": NUBAR_PROMPT,
    "delayed nu": NUBAR_DELAYED,
    "chi total": CHI_TOTAL,
    "total chi": CHI_TOTAL,
    "fission chi": CHI_TOTAL,
    "chi prompt": CHI_PROMPT,
    "chi delayed": CHI_DELAYED,
    "prompt chi": CHI_PROMPT,
    "delayed chi": CHI_DELAYED,
    "ela leg mom 1": ELASTIC_LEGENDRE_P1,
    "ela leg mom 2": ELASTIC_LEGENDRE_P2,
    "elastic law legendre p1": ELASTIC_LEGENDRE_P1,
    "elastic law legendre p2": ELASTIC_LEGENDRE_P2,
    "elastic law": ELASTIC_LEGENDRE_P1,
    "scatter law legendre p1": SCATTER_LEGENDRE_P1,
    "scatter law legendre p2": SCATTER_LEGENDRE_P2,
    "scatter law": SCATTER_LEGENDRE_P1,
    "scattering law legendre p1": SCATTER_LEGENDRE_P1,
    "scattering law legendre p2": SCATTER_LEGENDRE_P2,
    "scattering law": ELASTIC_LEGENDRE_P1,
    "inelastic law legendre p1": INELASTIC_LEGENDRE_P1,
    "inelastic law legendre p2": INELASTIC_LEGENDRE_P2,
    "inelastic law": INELASTIC_LEGENDRE_P1,
    "n,xn": N_XN_XS,
    "nxn": N_XN_XS,
    "n,alpha": N_ALPHA_XS,
    "nalpha": N_ALPHA_XS,
    "capture": CAPTURE_XS,
})

ERANOS_CHANNELS = {
                   "CAPTURE": CAPTURE_XS,
                   "FISSION": FISSION_XS,
                   "ELASTIC": ELASTIC_XS,
                   "INELASTIC": INELASTIC_XS,
                   "N,XN": N_XN_XS,
                   "NU": NUBAR_TOTAL,
                   }


def eranos_channel(label, isotope=None):
    """Return the ENDF channel matching an ERANOS perturbation label."""
    if label == "CAPTURE" and isotope == "B-10":
        return N_ALPHA_XS
    return ERANOS_CHANNELS[label]
