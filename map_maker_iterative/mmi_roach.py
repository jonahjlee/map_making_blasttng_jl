import numpy as np
from mmi_config import (ScanPass, slice_i_dict, pass_indices, dir_roach_dict,
                        dir_targ_dict, dir_master,
                        source_name, platescale, RoachID, kid_ref_dict, kid_max_dict, file_rejects_dict)
import mmi_data_lib as dlib
import mmi_map_lib as mlib


class Roach:
    """Object representation of BLAST-TNG ROACH.

    Provides access to roach data for a specified pass of RCW-92.
    """

    def __init__(self, roach_id: RoachID, scan_pass: ScanPass):
        self.roach_id = roach_id
        self.id = roach_id.value
        self.scan_pass: ScanPass = scan_pass

        self.dir_roach = dir_roach_dict[self.id]
        self.dir_targ = dir_targ_dict[self.id]
        self.kid_ref = kid_ref_dict[self.id]
        self.kid_max = kid_max_dict[self.id]
        self.file_rejects = file_rejects_dict[self.id]

        self.slice_i, self.slice_f = self._get_slice_interval()

        self._dat_targs = None
        self._Ff = None
        self._dat_align_indices = None
        self._dat_sliced = None

        self.x_az, self.y_el = self._az_el_offsets()
        self._x_um, self._y_um = mlib.offsetsTanProj(self.x_az, self.x_az, platescale)

        self.kids = self._load_kids()

    def _load_kids(self):
        # kids to use
        kids = dlib.findAllKIDs(self.dir_roach)  # all in dir_roach; sorted

        # remove unused channels
        kids = [kid for kid in kids if int(kid) <= self.kid_max]

        # KID rejects
        try:  # file might not exist
            kid_rejects = dlib.loadKidRejects(self.file_rejects)
            kids = [kid for kid in kids if kid not in kid_rejects]
        except FileNotFoundError:
            pass

        # move ref kid so it's processed first
        # this is last so it raises an error if our ref has been removed
        kids.remove(self.kid_ref)
        kids.insert(0, self.kid_ref)

        return kids

    def _get_slice_interval(self) -> tuple[int, int]:
        """Determines the starting and ending indices of the desired slice for this ROACH.

        Returns a tuple in the form (slice_i, slice_f).
        """

        if self.scan_pass == ScanPass.ALL:
            slice_i = slice_i_dict[self.id]
            slice_f = slice_i + pass_indices[ScanPass.ALL.value]
        else:
            slice_i  = slice_i_dict[self.id] + pass_indices[self.scan_pass.value]
            slice_f = slice_i + pass_indices[self.scan_pass.value + 1]

        return slice_i, slice_f

    def _az_el_offsets(self) -> tuple[np.ndarray, np.ndarray]:
        # detected source coordinates in az/el telescope frame
        source_azel = mlib.sourceCoordsAzEl(  # (az, el)
            source_name,
            self.dat_sliced['lat'], self.dat_sliced['lon'],
            self.dat_sliced['alt'], self.dat_sliced['time'])

        # generate x_az and y_el, the az/el offset tods
        return mlib.azElOffsets(
            source_azel,
            self.dat_sliced['az'],
            self.dat_sliced['el']
        )

    def _load_master_data(self):
        """Loads master data.

        The following instance attributes are loaded:
            - self._dat_sliced
            - self._dat_align_indices
        """
        dat_raw = dlib.loadMasterData(self.id, dir_master, self.dir_roach)

        # temporally align tods, rebin if necessary
        dat_aligned, self._dat_align_indices = dlib.alignMasterAndRoachTods(dat_raw)

        # slice tods to desired region (remove cal lamp)
        self._dat_sliced = {
            field: dat_aligned[field][self.slice_i:self.slice_f].copy()
            for field in dat_aligned}

    def _load_target_sweeps(self):
        """Loads target sweep data.

        The following instance attributes are loaded:
            - self._dat_targs
            - self._Ff
        """
        self._dat_targs, self._Ff = dlib.loadTargSweepsData(self.dir_targ)

    @property
    def dat_targs(self):
        if self._dat_targs is None:
            self._load_target_sweeps()
        return self._dat_targs

    @property
    def Ff(self):
        if self._Ff is None:
            self._load_target_sweeps()
        return self._Ff

    @property
    def dat_align_indices(self):
        if self._dat_align_indices is None:
            self._load_master_data()
        return self._dat_align_indices

    @property
    def dat_sliced(self):
        if self._dat_sliced is None:
            self._load_master_data()
        return self.dat_sliced


if __name__ == '__main__':
    my_roach = Roach(RoachID.ROACH_1, ScanPass.ALL)
    print(my_roach.dat_sliced.keys())
