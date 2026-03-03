"""This file contains the ULM class."""

from typing import Tuple

import numpy as np
from loguru import logger
from peasyTracker import SimpleTracker
from scipy.ndimage import maximum_filter
from scipy.signal import butter, convolve, lfilter
import torch

import torch.nn.functional as F
from ulm3d.loc.radial_symmetry_center_torch import radial_symmetry_center_3d_torch_batch
from ulm3d.utils.load_data import load_iq
from ulm3d.utils.matlab_tool import smooth

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class ULM:
    """
    The ULM class which contains ULM parameters and methods to apply the pipeline.

    Attributes:
        res (int): The hypothetical super-resolution factor.
        origin (int): Reconstruction grid origin in each direction.
        voxel_size (int): Voxel size in each direction [µm].
        z_dim (int): Dimension in the IQ where z is located [1,2,3].
        size (np.ndarray): Input size of raw beamformed data (IQ).
        scale (np.ndarray): Scaling for each dimension [mm mm mm s].
        svd_tresh (np.ndarray): The range of singular values preserved for SVD filter.
        filter_order (int): The order of the bandpass filter.
        filter_fs (float): Sampling frequency [Hz] (volumerate).
        filter_fc (float): Cutoff frequency for bandpass filter [Hz].
        filter_num (np.ndarray): Numerator (b) polynomials of the butterworth filter.
        filter_den (np.ndarray): Denominator (a) polynomials of the butterworth filter.
        filt_mode (str): filtering mode.
            filt_mode can be:
                - 'no_filter': no filter.
                - 'SVD': only SVD.
                - 'SVD_bandpass': SVD + bandpass.
        number_of_particles (int): The number of microbubbles to be localized in every frame.
        fwhm (np.ndarray): Size of the PSF for localization.
        min_snr (int): The minimum SNR value for a microbubble to be accepted [dB].
        patch_size (np.ndarray): The size of the 3D kernel where the local SNR is computed.
        nb_local_max (int): The maximum number of allowed microbubbles inside a patch (fwhm^3).
        max_gap_closing (int): The maximum number of frames that the microbubble is allowed to disappear and reappear (PeasyTracker parameter).
        max_linking_distance (int): Maximum linking distance between two frames to reject pairing, in super-resolved voxels. Should be between 20 to 40 SRvoxels (PeasyTracker parameter).
        min_length (int): The minimum number of frames a microbubble must be tracked for it to be accepted.

    Methods:
        -filtering: Filter the IQ data to remove tissue signal, tissue motion, and skull signal to enhance microbubble signals.
        -super_localization: Super-localization function to pinpoint the microbubble's position with sub-wavelength precision.
        -create_tracks: 3D Tracking function. The 3D tracking algorithm is a Python-adapted ('PeasyTracker') version of the open-source 'SimpleTracker' [Tinevez, J.-Y. et al. 2017].
    """

    def __init__(
        self,
        res: int,
        max_velocity: int,
        svd_values: list,
        filter_order: int,
        bandpass_filter: list,
        filt_mode: str,
        number_of_particles: int,
        nb_local_max: int,
        fwhm: list,
        min_snr: int,
        patch_size: tuple,
        min_length: int,
        max_gap_closing: int,
        z_dim: int,
        volumerate: int,
        voxel_size: list,
        origin: int,
        iq_files: list,
        input_var_name="",
        log = None,
        **kwargs,
    ) -> None:
        """To initialize the ULM class.

        Args:
            volumerate (int): Number of volumes acquired per second.
            z_dim (int): Dimension in the IQ where z is located [1,2,3].
            voxel_size (float): Voxel size in each direction [mm].
            origin (list): Reconstruction grid origin in each direction [mm].
            res (int): The hypothetical super-resolution factor.
            max_velocity (int): The max speed of the particle to track [mm/s]. (Converted to maximum linking distance).
            bandpass_filter (int): Cutoffs for the bandpass filter.
            svd_values (int): The singular values that will be kept after applying SVD filtering.
            number_of_particles (int): The number of microbubbles to be localized in every frame.
            min_length (int): The minimum number of frames a microbubble must be tracked for it to be accepted.
            filter_order (int): The order of the bandpass filter.
            filt_mode (str): Filtering mode.
                filt_mode can be:
                    - 'no_filter': no filter.
                    - 'SVD': only SVD.
                    - 'SVD_bandpass': SVD + bandpass.
            fwhm (list): Size of the PSF for localization in each direction.
            nb_local_max (int): The maximum number of allowed microbubbles inside a patch (fwhm^3).
            min_snr (int): The minimum SNR value for a microbubble to be accepted [dB].
            patch_size (np.ndarray): The size of the 3D kernel where the local SNR is computed.
            max_gap_closing (int): The maximum number of frames that the microbubble is allowed to disappear and reappear (PeasyTracker parameter).
            input_var_name (str): Name of the input variable when IQ are loaded.
            iq_files (list): List of IQ paths. Used to load an IQ to determine the shape.
        """
        print("ULM RECEIVED LOG =", log)
        if log is None:
            log = []

        self.log = log
        if "pipeline" in self.log:
            logger.info("Initializing ULM pipeline...")
        for key, value in kwargs.items():
            setattr(self, key, value)

        # Size parameters.
        self.res = res
        self.origin = np.array(origin)
        self.z_dim = z_dim

        iq = load_iq(iq_files[0], input_var_name)
        self.size = iq.shape
        self.scale = np.append(np.array(voxel_size), 1 / volumerate)

        # Filtering parameters.
        self.svd_tresh = np.array(svd_values, dtype=np.uint8)
        self.filter_order = filter_order

        if filt_mode == "SVD_bandpass":
            self.filter_fs = volumerate
            self.filter_fc = np.array(
                bandpass_filter,
                dtype=np.uint8,
            )
            self.filter_num, self.filter_den = butter(
                self.filter_order, self.filter_fc / (self.filter_fs / 2), "bandpass"
            )
            if "pipeline" in self.log:
                logger.debug(
                    f"Set bandpass filter: order {filter_order}, fc {self.filter_fc}"
                )

        self.filt_mode = filt_mode

        # Detection parameters.
        self.number_of_particles = number_of_particles
        self.fwhm = np.array(fwhm)
        self.min_snr = min_snr
        self.patch_size = np.array(patch_size)
        self.nb_local_max = nb_local_max

        # Tracking parameters.
        self.max_gap_closing = max_gap_closing
        if "pipeline" in self.log:
            logger.info(f"max_gap_closing: { self.max_gap_closing}")

        max_link_dist = max_velocity / volumerate / voxel_size[self.z_dim] * self.res
        self.max_linking_distance = np.round(max_link_dist)
        mld_mm = self.max_linking_distance / self.res * np.mean(self.scale[:3])
        if "pipeline" in self.log:
            logger.info(
                f"max_linking_distance_dist: {mld_mm} (~{mld_mm/np.mean(self.scale[:3])} voxel)"
            )
        self.min_length = min_length
        if "pipeline" in self.log:
            logger.info(f"min_track_len: { self.min_length}")

    def filtering(self, iq: np.ndarray) -> np.ndarray:

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        iq_t = torch.from_numpy(iq).to(device, non_blocking=True).to(torch.complex128)

        iq_shape = iq_t.shape
        T = iq_shape[-1]

        iq_t = iq_t.permute(3, 0, 1, 2).contiguous()  # (T, X, Y, Z)
        iq_t = iq_t.reshape(T, -1).T                  # (-1, T)

        cov = torch.matmul(torch.conj(iq_t.T), iq_t)

        u, s, vh = torch.linalg.svd(cov, full_matrices=False)

        v = torch.matmul(iq_t, u)

        k0 = self.svd_tresh[0] - 1
        k1 = self.svd_tresh[1]

        iq_filtered = torch.matmul(
            v[:, k0:k1],
            torch.conj(u[:, k0:k1]).T
        )

        iq_filtered = iq_filtered.T.reshape(T, *iq_shape[:-1])
        iq_filtered = iq_filtered.permute(1, 2, 3, 0).contiguous()

        iq_filtered = iq_filtered.cpu().numpy()

        if self.filt_mode == "SVD_bandpass":
            iq_filtered = lfilter(self.filter_num, self.filter_den, iq_filtered)

        return iq_filtered


    def super_localization(
        self,
        iq: np.ndarray,
        type_name: str = "float",
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Super-localization function to pinpoint the microbubble's position with sub-wavelength precision.

        Args:
            iq (np.ndarray): The filtered IQ data.
            type_name (str, optional): The type of the IQ matrix. Defaults to "float".

        Returns:
            np.ndarray: A structured array containing a matrix with the super-localization positions in sub-voxels. 
                        The fields are:
                - snr: The local SNR of the microbubble.
                - pos: The sub-wavelength position of the microbubble (sub-voxel).
                - frame_no: The index of the frame where the microbubble is located.
        """
        iq = np.abs(np.asarray(iq))
        mask, intensity = get_intensity_matrix(iq, self.fwhm, type_name)

        (
            local_contrast,
            final_mask,
            linear_ind_mb_detection,
            index_frames,
        ) = get_local_contrast(
            iq,
            mask,
            intensity,
            min_snr=self.min_snr,
            patch_size=self.patch_size,
        )

        # Verify if the number of detections exceeds the expected number of particles.
        # If there are too many detections, remove the particles with the lowest local contrast values.
        if np.size(index_frames) / self.size[-1] > self.number_of_particles:
            logger.debug(
                f"Too many detections, reducing to {self.number_of_particles} per frame."
            )
            for frame in range(self.size[-1]):
                idx_current_frame = np.unravel_index(
                    linear_ind_mb_detection[index_frames == frame],
                    local_contrast.shape,
                    order="F",
                )
                local_contrast_i = local_contrast[
                    idx_current_frame
                ]  # value for each frame.
                if np.size(local_contrast_i) > self.number_of_particles:
                    # Sort in descending order.
                    sort_idx_current_frame = np.argsort(local_contrast_i)[::-1]
                    # remove bubble i ind.
                    local_contrast_i[
                        sort_idx_current_frame[self.number_of_particles :]
                    ] = 0

                    # Keep only the linear index that matches the condition of number of particles.
                    linear_ind_mb_detection[index_frames == frame] = (
                        linear_ind_mb_detection[index_frames == frame]
                        * (local_contrast_i > 0)
                    )

        linear_ind_mb_detection = linear_ind_mb_detection[linear_ind_mb_detection != 0]
        logger.debug(f"{len(linear_ind_mb_detection)} maxima kept")
        ind_4D = np.unravel_index(
            linear_ind_mb_detection, local_contrast.shape, order="F"
        )
        list_snr = local_contrast[ind_4D]
        indices_4d = np.unravel_index(
            linear_ind_mb_detection,
            final_mask.shape,
            order="F",
        )
        index_mask = np.stack(indices_4d)
        index_frames = indices_4d[-1]

        index_frames = np.sort(index_frames)
        ind_t = np.sort(np.argsort(index_frames))
        index_mask = index_mask[:, ind_t]

        # Creating FWHM models.
        vectfwhm = [None] * 3
        for i, h_fwhm in enumerate(np.ceil(self.fwhm / 2).astype(int)):
            v_fwhm = np.arange(-h_fwhm, h_fwhm + 1).astype(np.int64)
            v_shape = [1] * 3
            v_shape[i] = -1
            vectfwhm[i] = np.reshape(v_fwhm, v_shape)

        # Localization.
        intensity_center = np.zeros((index_mask.shape[1]), dtype=type_name)
        pos = np.zeros_like(index_mask[:3, :], dtype=np.float32)
        rois = []
        rois_indices = []
        
        # Localization (BATCHED)
        n_scats = index_mask.shape[1]
        
        rois = []
        rois_iscat = []
        # Pure extraction
        for iscat in range(n_scats):
            intensity_roi = np.absolute(
                iq[
                    index_mask[0, iscat] + vectfwhm[0],
                    index_mask[1, iscat] + vectfwhm[1],
                    index_mask[2, iscat] + vectfwhm[2],
                    index_mask[3, iscat],
                ]
            )
            rois.append(intensity_roi.astype(np.float32, copy=False))
            rois_iscat.append(iscat)

        if len(rois) == 0:
            logger.warning("0 microbubbles located!")
            return False, None

        # Massive upload to GPU
        rois_torch = torch.from_numpy(np.stack(rois, axis=0)).to(DEVICE, non_blocking=True)
        
        # Local maximum filtering in BATCH on GPU
        rois_torch_unsqueeze = rois_torch.unsqueeze(1)
        mask_loc_gpu = F.max_pool3d(rois_torch_unsqueeze, kernel_size=3, stride=1, padding=1)
        
        # Counting maxima per patch
        is_max = (rois_torch_unsqueeze == mask_loc_gpu).view(n_scats, -1)
        max_counts = is_max.sum(dim=1)
        
        # Boolean mask of valid patches
        valid_mask = max_counts <= self.nb_local_max
        
        # Keep only valid patches for radial symmetry calculation
        rois_torch_valid = rois_torch[valid_mask]
        valid_indices = torch.nonzero(valid_mask).squeeze(-1).cpu().numpy()
        rois_iscat = np.array(rois_iscat)[valid_indices]

        if len(rois_torch_valid) == 0:
            logger.warning("0 microbubbles valid after max filter!")
            return False, None

        # Call radial symmetry function on valid subset
        with torch.no_grad():
            sub_pos_batch = radial_symmetry_center_3d_torch_batch(rois_torch_valid, log=self.log)  # (B,3)
        
        sub_pos_batch = sub_pos_batch.cpu().numpy()
        rois_iscat = np.array(rois_iscat, dtype=np.int64)
        order = np.argsort(rois_iscat)
        
        rois_iscat = rois_iscat[order]
        sub_pos_batch = sub_pos_batch[order]
        
        index_mask = index_mask[:, rois_iscat]
        
        pos = index_mask[:3, :].astype(np.float32)
        pos += sub_pos_batch.T
        
        intensity_center = list_snr[rois_iscat].astype(type_name)
        new_struct = [
            (intensity_center[i], pos[:, i], index_mask[3, i])
            for i in range(pos.shape[1])
        ]
        
        structured_localizations = np.array(
            new_struct,
            dtype=[
                ("snr", float),
                ("pos", float, 3),
                ("frame_no", int),
            ],
        )
        return structured_localizations

    def create_tracks(self, localizations: np.ndarray) -> np.ndarray:
        """
        3D Tracking function. The 3D tracking algorithm is a Python-adapted ('PeasyTracker') version of the open-source 'SimpleTracker' [Tinevez, J.-Y. et al. 2017].

        Args:
            localizations (np.ndarray): The sub-wavelength localization matrix.

        Returns:
            Tuple[np.ndarray, np.ndarray]: A tuple containing structured arrays with tracking information.
                                           Interpolated tracks at index 0 and raw tracks at index 1.

            The fields for interpolated tracks are:
                - pos: The interpolated sub-wavelength position of the microbubble [mm].
                - speed: The interpolated speed of the microbubble [mm/s]
                - time: The interpolated time [s].
                - track_ind: The index of the track.

            The fields for raw tracks are:
                - pos: The sub-wavelength position of the microbubble [dimension].
                - time: The frame index.
                - track_ind: The index of the track.
        """
        pitch = np.mean(self.scale[:3])
        if "tracking" in self.log:
            logger.debug(f"Average voxel pitch {pitch} ({self.scale[:3]})")

        # Convert localizations from pixel to [mm], frame to time in seconds.
        localizations["pos"] = localizations["pos"] * self.scale[:3] + self.origin

        # Tracking
        if "tracking" in self.log:
            logger.debug(f"Start SimpleTracker on {len(localizations)}")
        max_linking_distance_dist = self.max_linking_distance / self.res * pitch
        if "tracking" in self.log:
            logger.debug(
                f"max_linking_distance_dist: {max_linking_distance_dist} (~{max_linking_distance_dist/pitch} voxel)"
            )
        tracked_loc = SimpleTracker(
            data=localizations,
            max_linking_dist=max_linking_distance_dist,
            max_gap_closing=self.max_gap_closing,
            min_track_len=self.min_length,
        )
        # Output from SimpleTracker: ['pos', 'frame_no', 'track_no']
        track_ids = np.unique(tracked_loc["track_no"])
        # Remove localizations not associated with tracks.
        track_ids = track_ids[track_ids >= 0]

        # Init list for structured arrays. Collect track indexes.
        raw_tracks = np.zeros(
            0,
            dtype=[
                ("pos", float, 3),
                ("time", float),
                ("track_ind", int),
            ],
        )
        interp_tracks = np.zeros(
            0,
            dtype=[
                ("pos", float, 3),
                ("velocity", float, 3),
                ("time", float),
                ("track_ind", int),
            ],
        )
        if "tracking" in self.log:  
            logger.debug(f"{len(track_ids)} tracks found.")

        # Use empty Python lists
        all_raw_tracks = []
        all_interp_tracks = []

        # Pre-extract columns to avoid accessing the structured dictionary in each iteration
        t_pos = tracked_loc["pos"]
        t_frame = tracked_loc["frame_no"]
        t_track_no = tracked_loc["track_no"]

        for track_id in track_ids:
            # Create a boolean mask to isolate the bubble
            mask = (t_track_no == track_id)
            
            raw_track, interp_track = clean_and_interpolate_track(
                pos=t_pos[mask],
                index_frames=t_frame[mask],
                interp_factor=1 / self.res,
                scale=self.scale,
                track_id=track_id,
            )
            
            # 2. .extend() adds list elements without global memory reallocation
            all_raw_tracks.extend(raw_track)
            all_interp_tracks.extend(interp_track)

        # 3. Final one-shot conversion (Zero-Copy overhead)
        raw_tracks_dtype = [
            ("pos", float, 3),
            ("time", float),
            ("track_ind", int),
        ]
        interp_tracks_dtype = [
            ("pos", float, 3),
            ("velocity", float, 3),
            ("time", float),
            ("track_ind", int),
        ]
        
        raw_tracks_np = np.array(all_raw_tracks, dtype=raw_tracks_dtype)
        interp_tracks_np = np.array(all_interp_tracks, dtype=interp_tracks_dtype)

        return [interp_tracks_np, raw_tracks_np]


def get_intensity_matrix(iq, fwhm, type_name) -> np.ndarray:
    """Get the intensity of a voxel when a local maximum is located.

    Args:
        iq (np.ndarray): The filtered IQ data.
        fwhm (np.ndarray): Size of the PSF for localization.
        type_name (str, optional): The type of the IQ matrix. Defaults to "float".

    Returns:
        np.ndarray: An intensity matrix same size as iq. Contains the intensity value of each voxel when a local maximum is found.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    iq_tensor = torch.from_numpy(iq).to(device, dtype=torch.float32)

    half_fwhm = np.ceil(1 + fwhm / 2).astype(int)
    
    iq_reduced = torch.zeros_like(iq_tensor)
    iq_reduced[
        half_fwhm[0] : -half_fwhm[0],
        half_fwhm[1] : -half_fwhm[1],
        half_fwhm[2] : -half_fwhm[2],
        :,
    ] = iq_tensor[
        half_fwhm[0] : -half_fwhm[0],
        half_fwhm[1] : -half_fwhm[1],
        half_fwhm[2] : -half_fwhm[2],
        :,
    ]


    iq_batched = iq_reduced.permute(3, 0, 1, 2).unsqueeze(1)

    max_filtered = F.max_pool3d(iq_batched, kernel_size=3, stride=1, padding=1)

    mask_tensor = (max_filtered == iq_batched) & (iq_batched > 0)

    mask_tensor = mask_tensor.squeeze(1).permute(1, 2, 3, 0)
    
    intensity_tensor = mask_tensor * iq_tensor

    return mask_tensor.cpu().numpy(), intensity_tensor.cpu().numpy()



def get_local_contrast(
    iq: np.ndarray,
    mask: np.ndarray,
    intensity_matrix: np.ndarray,
    min_snr: float,
    patch_size: np.array,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Retrieves the local contrast of local maxima to detect microbubbles.

            Args:
                iq (np.ndarray): The filtered IQ data.
                mask (np.ndarray): The binary mask where local maxima are found.
                intensity_matrix (np.ndarray): The associated value of local maxima.
                min_snr (float): min SNR to keep
                patch_size (np.ndarray): size of patch for SNR computation
                device (str): Device for computation ('cuda' or 'cpu')

    Returns:
                Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]: ...
    """

    # Create convolution kernel
    mat_conv = np.ones(patch_size, dtype=np.float32)
    tt = np.arange(1, mat_conv.shape[0] + 1) - (mat_conv.shape[0] + 1) / 2
    [meshz, meshx, meshy] = np.meshgrid(tt, tt, tt)
    meshr = np.sqrt(meshz**2 + meshx**2 + meshy**2)
    mat_conv[meshr < np.sqrt(3) + 0.01] = 0.2  # 1st neighbours voxels.
    mat_conv[meshr < 1] = 0  # center voxel.
    mat_conv = mat_conv / np.sum(mat_conv)

    # Convert kernel to PyTorch tensor for conv3d: (out_channels, in_channels, D, H, W)
    kernel = torch.from_numpy(mat_conv).float().to(device, non_blocking=True)
    kernel = kernel.unsqueeze(0).unsqueeze(0)  # Shape: (1, 1, D, H, W)

    # Calculate clutter
    clutter = iq - intensity_matrix

    # Convert to PyTorch tensor
    if clutter.dtype != np.float32:
        clutter = clutter.astype(np.float32)
    clutter_tensor = torch.from_numpy(clutter).to(device, non_blocking=True)
    clutter_tensor = clutter_tensor.permute(3, 0, 1, 2).unsqueeze(1)

    # Padding for 'same' mode
    pad = [s // 2 for s in patch_size]
    padding = (pad[2], pad[2], pad[1], pad[1], pad[0], pad[0])
    clutter_padded = F.pad(clutter_tensor, padding, mode='replicate')

    # 3D convolution on GPU
    clutter_conv = F.conv3d(clutter_padded, kernel)

    # Reshape to original form: (T, 1, X, Y, Z) -> (X, Y, Z, T)
    clutter_conv = clutter_conv.squeeze(1).permute(1, 2, 3, 0)

    # Send intensity_matrix and mask to GPU
    intensity_tensor = torch.from_numpy(intensity_matrix).to(device, non_blocking=True)
    mask_tensor = torch.from_numpy(mask).to(device, non_blocking=True)

    # Calculate local contrast (on GPU)
    local_contrast = intensity_tensor / (clutter_conv + 1e-12)
    
    # Masking and dB conversion on GPU
    local_contrast[~mask_tensor] = 0.0
    local_contrast = 20 * torch.log10(local_contrast + 1e-12)
    
    logger.trace(f"Apply min SNR threshold at {min_snr}dB")
    local_contrast[local_contrast < min_snr] = 0

    final_mask = local_contrast > 0
    # Replace NaNs with 0 (torch.isnan)
    final_mask[torch.isnan(final_mask)] = False
    final_mask = final_mask * intensity_tensor

    # Bring ONLY the final result back to CPU
    local_contrast = local_contrast.cpu().numpy()
    final_mask = final_mask.cpu().numpy()

    final_mask_flatten_F = np.ndarray.flatten(final_mask, order="F")
    linear_ind_mb_detection = np.nonzero(final_mask_flatten_F)[0]
    logger.debug(f"{len(linear_ind_mb_detection)} maxima found")

    _, _, _, index_frames = np.unravel_index(
        linear_ind_mb_detection, final_mask.shape, order="F"
    )
    return local_contrast, final_mask, linear_ind_mb_detection, index_frames

def get_curvilinear_abscissa(m: np.ndarray, axis: int = 0):
    m = np.diff(m, axis=axis)
    d_ca = np.linalg.norm(m, ord=2, axis=1 - axis)
    ca = np.concatenate(([0], np.cumsum(d_ca)))
    return ca


def clean_and_interpolate_track(
    pos: np.array,
    scale: np.array,
    index_frames: np.array,
    interp_factor: float,
    smooth_window: int = 5,
    track_id: int = 0,
):
    """
    Clean and interpolation function to process raw microbubble tracks. Performs smoothing and interpolation.

    Args:
        pos (np.ndarray) [Nx3]: raw microbubble positions.
        scale (np.ndarray) [4]: dimension scale [space, space, space, time].
        index_frames (np.ndarray) [Nx1]: frame index.
        interp_factor (float): sub scale interpolation factor.
        smooth_window (int): smoothing window on raw positions.
        track_id (int, optional): index of the track.

    Returns:
        list of raw data [(position, frame_index, track_id)...]
        list of interpolated data [(position, velocity, frame_index, track_id)...]
    """

    # Smooth tracks (pos : [N x 3])
    pos = smooth(pos, window=smooth_window)

    # Calculate curvilinear abscissa along track
    ca = get_curvilinear_abscissa(pos)
    ca_interp = np.arange(ca[0], ca[-1], interp_factor * np.min(scale[:3]))

    # Calculate interpolated positions and smoothing
    pos_i = np.zeros([len(ca_interp), pos.shape[1]], dtype=pos.dtype)
    for i in range(pos_i.shape[1]):
        pos_i[:, i] = np.interp(ca_interp, ca, pos[:, i])
    pos_i = smooth(pos_i, scale[i] * 2)

    # Calculate curvilinear abscissa along interpolated track
    ca_i = get_curvilinear_abscissa(pos_i)

    tl = index_frames * scale[-1]  # Track timeline
    tl_i = np.interp(ca_i / ca_i[-1], ca / ca[-1], tl)
    dt_line = np.diff(tl_i)

    # Calculate interpolated velocities in [distance/s]
    vel = np.diff(pos_i, axis=0) / np.expand_dims(dt_line, 1)
    vel = np.row_stack([vel[0, :], vel])

    raw_list = []
    for i in range(pos.shape[0]):
        raw_list.append(
            (
                pos[i, :].T,
                index_frames[i],
                track_id,
            )
        )
    interp_list = []
    for i in range(pos_i.shape[0]):
        interp_list.append(
            (
                pos_i[i, :].T,
                vel[i, :].T,
                tl_i[i],
                track_id,
            )
        )
    return [raw_list, interp_list]