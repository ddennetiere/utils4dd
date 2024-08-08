# -*- coding: utf-8 -*-

"""
This file is part of the Superflat Evaluation Toolkit

This file provides an interface to the data structures of the Zygo.dat file format

The `DatFile` class stores the file metadata and provides a fast access to the stored phase image using
DatFile.getHeight

Useful metadata are kept in the `DatFile.header` member and its `Header.BEattributes` member

"""

__author__ = "François Polack"
__copyright__ = "2022, Superflat-PCP project"
__email__ = "superflat-pcp@synchrotron-soleil.fr"
__version__ = "0.1.0"

__pdoc__ = {}

import tkinter
from tkinter import filedialog
import logging
import numpy as np
import struct
from pathlib import Path
from ctypes import sizeof, BigEndianStructure, LittleEndianStructure, c_float, c_char, c_byte, c_ubyte, c_short, \
    c_ushort, c_int, c_uint

ZygoSoftwareType = ("Unknown", "MetroPro", "MetroBasic", "Diagnostic")
acqMode = ("phase", "fringe", "scan")
logger = logging.getLogger(__name__)


class ZygoSoftwareVersion(BigEndianStructure):
    _pack_ = 2
    _fields_ = [("Major", c_short),
                ("Minor", c_short),
                ("Bug", c_short)]

    def __str__(self):
        return f'{self.Major}.{self.Minor}.{self.Bug}'


class ZygoSoftwareInfo(BigEndianStructure):
    _pack_ = 2
    _fields_ = [("Type", c_short),  # The softwares are: unknown or MX (0), MetroPro (1), MetroBASIC (2), and d2bug (3).
                ("Date", c_char * 30),
                ("Version", ZygoSoftwareVersion)]


class ZygoBEattributes(BigEndianStructure):
    """  Attributes related to interferogram recording\n
        These attributes are stored in Big-endian order\n

        Only the most useful attributes are documented below.
        See the source code for a complete documentation
        """
    _pack_ = 2
    _fields_ = [("TimeStamp", c_int),
                # The date and time the data was measured. Number of seconds since 0:00:00 January 1, 1970*/
                ("Comment", c_char * 82),
                # User-entered remark line. The value is truncated to 81 characters when the data file is saved.*/
                ("Source", c_short),
                # ZygoSourceAttr	Indicates how the data was acquired or generated.Default=0= Measured*/
                ("IntfScaleFactor", c_float),  # The interferometric scale factor is the number of waves per fringe.*/
                ("WavelengthIn", c_float),
                # The wavelength, in meters, at which the interferogram was measured. Present */
                ("NumericAperture", c_float),  # Equals 1 / (2 * f-number). f/2.8 asumed */
                ("ObliquityFactor", c_float),
                # Phase correction factor required when using a Mirau objective on a microscope. A value of 1.0
                # indicates no correction.*/
                ("Magnification", c_float),  # Reserved for future use.*/
                ("LateralResolution", c_float),
                # Lateral resolving power of a camera pixel in meters/pixel. A value of 0 means that the value is
                # unknown.*/
                ("AcquireType", c_short),  # ZygoAcquireType	Indicates the setting of the Acquisition Mode control.*/
                ("IntensityAverageCount", c_short),
                # The number of intensity averages performed. Values of 0 or 1 indicate no averaging.*/
                ("PZTCal", c_short),
                # Boolean	Indicates whether or not the modulation amplitude was automatically adjusted during
                # acquisition.*/
                ("PZTGainTolerance", c_short),  # Specifies the PZT error range if PZT calibration was adjusted.*/
                ("PZTGain", c_short),  # Specifies the modulation amplitude value used during data acquisition.*/
                ("PartThickness", c_float),
                # The thickness, in meters, of the part measured. AGCBoolean Indicates whether or not automatic gain
                # control was performed during data acquisition.*/
                ("AGC", c_short),  # boolean : whether or not automatic gain control was active during data acquisition.
                ("TargetRange", c_float),  # The acceptable tolerance limits of the light intensity used during AGC.*/
                ("RadCrvMeasSeq", c_short),  # Undocumented */
                ("MinMod", c_int),  # Minimum value of modulation needed to calculate a phase value.*/
                ("MinModCount", c_int),  # Minimum number of data points required to pass MinMod criteria during AGC.*/
                ("PhaseRes", c_short),
                # ZygoResolution	The resolution of the phase data points. LowRes means that each fringe is
                # represented by 4096 counts. HighRes means that each fringe is represented by 32768 counts.
                # SuperRes, each fringe is 131072 counts*/
                ("MinAreaSize", c_int),  # Minimum number of contiguous data points required for a valid region.*/
                ("DisconAction", c_short),
                # ZygoDisconAction	Indicates the action taken when the system encounters discontinuities in phase
                # data.*/
                ("DisconFilter", c_float),
                # Specifies the degree to which discontinuities were removed when DisconAction was filtered.*/
                ("ConnectionOrder", c_short),
                # ZygoConnectionOrder	Order in which separate regions of phase data are processed.*/
                ("DataSign", c_short),  # Sign of the data. 0 is normal, 1 is inverted.*/
                ("CameraWidth", c_short),  # The width (columns) of the usable camera field in pixels. MINT default*/
                ("CameraHeight", c_short),  # The height (rows) of the usable camera field in pixels.*/
                ("SystemType", c_short),
                # ZygoSystemType	Indicates the system used to make the measurement. Default = Unknown*/
                ("SystemBoard", c_short),
                # Indicates which system board was used when the data measurement was taken. Valid values range from
                # 0 to 7.*/
                ("SystemSerial", c_short),  # The serial number of the instrument.*/
                ("InstrumentID", c_short),  # The instrument unit number. Valid values range from 0 to 7.*/
                ("ObjectiveName", c_char * 12),
                # For microscopes, this is the objective in use. For the GPI, this is the aperture in use. For
                # generated data, this is blank. The value is truncated to 11 characters when the data file is saved.*/
                ("PartName", c_char * 40),
                # User-entered identified for the part measured. The value is truncated to 39 characters when the
                # data file is saved.*/
                ("CodeVType", c_short),
                # ZygoCodeVType	Indicates whether the phase data represents a wavefront or a surface. This is only
                # used by the CODE V program.  default= surface is ambiguous because recrded values are WF but
                # surface is obtained by application of IntfScaleFactor*/
                ("PhaseAverageCount", c_short),  # The number of phase averages performed.*/
                ("SubtractSystemError", c_short),
                # Boolean	Indicates whether or not the system error was subtracted from the phase data.*/
                ("unused00", c_ubyte * 16),
                ("PartSerialNumber", c_char * 40),
                # User-entered serial number for the part. The value is truncated to 39 characters when the data file
                # is saved.*/
                ("RefractiveIndex", c_float),
                # Index of refraction specified by the user. This is used only in the calculation of corner cube
                # dihedral angles.*/
                ("RemoveTiltBias", c_short),
                # Indicates whether or not the tilt bias was removed from the phase data.*/
                ("RemoveFringes", c_short),
                # Boolean	Indicates whether or not fringes were removed from the intensity data.*/
                ("MaxAreaSize", c_int),
                # Maximum number of contiguous data points required for a valid phase data region. Any larger regions
                # are deleted.*/
                ("SetupType", c_short),  # Identifies the different setup used by the Angle Measurement Application.*/
                ("Wrapped", c_short),  # Boolean  */
                ("PreConnectFilte", c_float),  # The amount of filtering done to wrapped phase data.*/
                ("WavelengthIn2", c_float),
                # The value from the Wavelength-In 2 control, if relevant to the measurement.*/
                ("WavelengthFold", c_short),  # Boolean	 Indicates whether or not two wavelength folding is on.*/
                ("WavelengthIn1", c_float),
                # The value from the Wavelength-In 1 control, if relevant to the measurement.*/
                ("WavelengthIn3", c_float),
                # The value from the Wavelength-In 3 control, if relevant to the measurement.*/
                ("WavelengthIn4", c_float),
                # The value from the Wavelength-In 4 control, if relevant to the measurement.*/
                ("WavelengthSelect", c_char * 8),
                # The value of the Wavelength Select control, if relevant to the measurement. The value is truncated
                # to 7 characters when the data file is saved.*/
                ("FDARes", c_short),
                # ZygoFdaRes	Indicates the setting of the FDA Res control for a measurement when the Acquisition
                # Mode control is set to Scan.*/
                ("ScanDescription", c_char * 20),
                # Description of the scan for a measurement when the Acquisition Mode control is set to Scan.
                # The value is truncated to 19 characters when the data file is saved.
                # A sample string is “5 um bipolar”.*/
                ("NFiducials", c_short),  # The number of fiducials defined.*/
                ("FiducialX", c_float * 7),
                # The X component of the fiducial’s coordinates specified by Index. Index can range from 1 to 7.
                # The coordinate system is the camera’s and (0,0) is in the upper left corner of the video monitor.*/
                ("FiducialY", c_float * 7),
                # The Y component of the fiducial’s coordinates specified by Index. Index can range from 1 to 7.
                # The coordinate system is the camera’s and (0,0) is in the upper left corner of the video monitor.*/
                ("PixelWidth", c_float),
                # The effective camera pixel width (spacing) in meters. Affects the scaling of Mask Editor figures.
                # If zero, default values are assumed. Not the same as LateralResolution.*/
                ("PixelHeight", c_float),
                # The effective camera pixel height (spacing) in meters. Affects the scaling of Mask Editor figures.
                # If zero, default values are assumed. Not the same as LateralResolution.*/
                ("ExitPupilDiameter", c_float),  # The exit pupil diameter setting used during data acquisition.*/
                ("LightLevelPercent", c_float)]  # The light level percent used during data acquisition.*/


__pdoc__[
    'ZygoBEattributes.TimeStamp'] = "(`int`)  The date and time the data was measured. " \
                                    "Number of seconds since 0:00:00 January 1, 1970"
__pdoc__[
    'ZygoBEattributes.IntfScaleFactor'] = "(`float`)  The interferometric scale factor is the number of waves per " \
                                          "fringe"
__pdoc__[
    'ZygoBEattributes.WavelengthIn'] = "(`float`)  The wavelength, in meters, at which the interferogram was measured."
__pdoc__[
    'ZygoBEattributes.ObliquityFactor'] = "(`float`)  Phase correction factor required when using a Mirau objective " \
                                          "on a microscope. A value of 1.0 indicates no correction."
__pdoc__[
    'ZygoBEattributes.LateralResolution'] = "(`float`)  Lateral resolving power of a camera pixel in meters/pixel. " \
                                            "A value of 0 means that the value is unknown."
__pdoc__[
    'ZygoBEattributes.PhaseRes'] = "(`short`)  The resolution of the phase data points. LowRes means that each fringe "\
                                   "is represented by 4096 counts. HighRes means that each fringe is represented " \
                                   "by 32768 counts. SuperRes, each fringe is 131072 counts"
__pdoc__[
    'ZygoBEattributes.CameraWidth'] = "(`short`)  The width (columns) of the usable camera field in pixels. " \
                                      "MINT default"
__pdoc__['ZygoBEattributes.CameraHeight'] = "(`short`)  The height (rows) of the usable camera field in pixels."


class ZygoDataLocation(LittleEndianStructure):
    """       ZygoDataLocation structure is stored in little-endian byte order

            Used by `ZygoLEattributes` only
    """
    _pack_ = 4
    _fields_ = [
        ("XPosition", c_float),  # little-endian The position of the X-axis when the data was measured in inches.*/
        ("YPosition", c_float),  # little-endian The position of the Y-axis when the data was measured in inches.*/
        ("ZPosition", c_float),  # little-endian The position of the Z-axis when the data was measured in inches.*/
        ("XRotation", c_float),
        # little-endian The rotation of the X-axis (Roll) when the data was measured in degrees.*/
        ("YRotation", c_float),
        # little-endian The rotation of the Y-axis (Pitch) when the data was measured in degrees.*/
        ("ZRotation",
         c_float)]  # little-endian The rotation of the Z-axis (Theta) when the data was measured in degrees.*/


class ZygoLEattributes(LittleEndianStructure):
    """ These  attributes are set-up and used by Zygo software. They are stored little-endian unless specified

        Undocumented for a large part ; see source code for more information
    """
    _pack_ = 1
    _fields_ = [
        ("CoordinateState", c_int),
        # little-endian;  Indicates whether or not the coordinates in the Coordinates property are valid.
        # 0 indicates invalid coordinates.*/
        ("Coordinates", ZygoDataLocation),  # Coordinates of the stage during data acquisition.*/
        ("CoherenceMode", c_short),  # ZygoCoherenceMode	The coherence mode setting used during data acquisition.*/
        ("SurfaceFilter", c_short),  # ZygoSurfaceFilter	The surface filter setting used during data acquisition.*/
        ("SystemErrorFileName", c_char * 28),
        # User-entered name of the file containing the system error data. The value is truncated to 27 characters
        # when the data file is saved.*/
        ("ZoomDescription", c_char * 8),
        # Value of the image zoom used during data acquisition. The value is truncated to 7 characters when the data
        # file is saved.*/
        ("AplhaPart", c_float),  # undocumented*/
        ("BetaPart", c_float),  # undocumented*/
        ("DistancePart", c_float),  # undocumented*/
        ("CameraSplitLocationX", c_short),  # undocumented*/
        ("CameraSplitLocationY", c_short),  # undocumented*/
        ("CameraSplitTransX", c_short),  # undocumented*/
        ("CameraSplitTransY", c_short),  # undocumented*/
        ("MaterialA", c_char * 24),  # The value is truncated to 23 characters when the data file is saved.*/
        ("MaterialB", c_char * 24),  # The value is truncated to 23 characters when the data file is saved.*/
        ("CameraSplitUnused", c_short),  # undocumented*/
        ("unused01", c_ubyte * 2),  # 2 bytes unused
        ("DMICenterX", c_float),  # undocumented*/
        ("DMICenterY", c_float),  # undocumented*/
        ("SphDistCorr", c_short),  # Boolean undocumented*/
        ("unused02", c_ubyte * 2),  # 2 bytes unused
        ("SphDistPartNa", c_float),  # undocumented*/
        ("SphDistPartRadius", c_float),  # undocumented*/
        ("SphDistCalNa", c_float),  # undocumented*/
        ("SphDistCalRadius", c_float),  # undocumented*/
        ("SurfaceType", c_short),  # undocumented*/
        ("AcSurfaceType", c_short),  # undocumented*/
        ("ZPosition", c_float),  # undocumented*/
        ("PowerMultiplier", c_float),  # undocumented*/
        ("FocusMultiplier", c_float),  # undocumented*/
        ("RadCrvFocusCalFactor", c_float),  # undocumented*/
        ("RadCrvPowerCalFactor", c_float),  # undocumented*/
        ("FtpLeftPosition", c_float),  # undocumented*/
        ("FtpPitchPosition", c_float),  # undocumented*/
        ("FtpRightPosition", c_float),  # undocumented*/
        ("FtpRollPosition", c_float),  # undocumented*/
        ("MinModPercent", c_float),  # Min Mod as a percent*/
        ("MaxIntensity", c_int),  # intensity from frame grabber*/
        ("RingOfFire", c_short),  # as Boolean*/
        ("unused03", c_ubyte),  # 2 bytes unused
        ("RcOrientation", c_char),  # as ZygoOrientation*/
        ("RcDistance", c_float),  # undocumented*/
        ("RcAngle", c_float),  # undocumented*/
        ("RcDiamete", c_float),  # undocumented*/
        ("RemoveFringesMode", c_short),  # as BIG-endian Boolean	*/
        ("unused04", c_ubyte),  # 1 byte unused
        ("FtpsiPhaseResolution", c_ubyte),  # as ZygoFtpsiPhaseRes*/
        ("FramesAcquired", c_short),  # undocumented*/
        ("CavityType", c_short),  # as ZygoZavityType*/
        ("CameraFrameRate", c_float),  # undocumented*/
        ("TuneRange", c_float),  # Optical frequency excursion (Hz)*/
        ("CalPixelLocationX", c_short),  # X coordinate of cal pixel in camera coordinates*/
        ("CalPixelLocationY", c_short),  # Y coordinate of cal pixel in camera coordinates*/
        ("TestCalPointCount", c_short),  # undocumented*/
        ("ReferenceCalPointCount", c_short),  # undocumented*/
        ("TestCalPointX", c_float * 2),  # undocumented*/
        ("TestCalPointY", c_float * 2),  # undocumented*/
        ("ReferenceCalPointX", c_float * 2),  # undocumented*/
        ("ReferenceCalPointY", c_float * 2),  # undocumented*/
        ("TestCalPixelOpd", c_float),  # undocumented*/
        ("ReferenceCalPixelOpd", c_float),  # undocumented*/
        ("SystemSerial2", c_int),  # undocumented*/
        ("FlashphaseDCMask", c_float),  # undocumented*/
        ("FlashPhaseAliasMask", c_float),  # undocumented*/
        ("FlashPhaseFilter", c_float),  # undocumented*/
        ("ScanDirection", c_ubyte),  # as ZygoScanDirection*/
        ("unused05", c_ubyte),  # 1 byte unused
        ("FDAPreFilter", c_short),  # as BIG-endian integer*/
        ("unused06", c_ubyte * 4),  # 4 bytes unused
        ("FtpsiResolutionFactor", c_float),  # undocumented*/
        ("unused07", c_ubyte * 8)]  # 8 bytes unused


#  File header structure
class Header(BigEndianStructure):
    """
        Common part of the header of all Zygo.dat files

        It consist of 834 bytes, which are always read. Data are stored in big-endian byte order, unless specified
    """

    _pack_ = 2
    _fields_ = [("MagicNumber", c_uint),  # format validation number
                ("HeaderFormat", c_short),  # format type number
                ("HeaderSize", c_int),  # size of this header plus attribute structure
                ("SoftInfo", ZygoSoftwareInfo),  # Software date, type and version
                ("IntensOriginX", c_short),
                ("IntensOriginY", c_short),
                ("IntensWidth", c_short),
                ("IntensHeight", c_short),
                ("IntensNumBuckets", c_short),
                ("IntensRange", c_short),
                ("IntensNumBytes", c_int),
                ("PhaseOriginX", c_short),
                ("PhaseOriginY", c_short),
                ("PhaseWidth", c_short),
                ("PhaseHeight", c_short),
                ("PhaseNumBytes", c_int),
                ("BEattributes", ZygoBEattributes),
                ("LEattributes", ZygoLEattributes)]


__pdoc__[
    'Header.MagicNumber'] = "(`unsigned int`)   Indicates whether or not the file is a valid MetroPro file. " \
                            "\n\nThe file is valid only if the value is –2011495569 (881B036F hex)"
__pdoc__['Header.HeaderFormat'] = "(`short`)   Indicates the format of the header. This can be 1, 2 or 3."
__pdoc__['Header.HeaderSize'] = "(`ZygoSoftwareInfo`)   Date, type and version of the software which created the file"
__pdoc__[
    'Header.SoftInfo'] = "(string)   The size of the header in bytes. This should be 834 for header format 1 and 2, " \
                         "and 4096 for format 3."
__pdoc__[
    'Header.IntensOriginX'] = "(short)   The X component of the origin of the intensity data matrix. " \
                              "\n\nThe coordinate system is the camera’s and (0,0) is in the upper left corner" \
                              " of the video monitor. "
__pdoc__[
    'Header.IntensOriginY'] = "(short)   The Y component of the origin of the intensity data matrix. " \
                              "\n\nThe coordinate system is the camera’s and (0,0) is in the upper left corner " \
                              "of the video monitor. "
__pdoc__[
    'Header.IntensWidth'] = "(short)   The width (number of columns) of the intensity data matrix. " \
                            "\n\nIf no intensity data matrix is present, then this value is zero."
__pdoc__[
    'Header.IntensHeight'] = "(short)   The height (number of rows) of the intensity data matrix. " \
                             "\n\nIf no intensity data matrix is present, then this value is zero."
__pdoc__[
    'Header.IntensNumBuckets'] = "(short)   The number of buckets of intensity data that are stored. " \
                                 "\n\nIf no intensity data matrix is present, then this value is zero."
__pdoc__[
    'Header.IntensRange'] = "(short)   Number of bytes of intensity data stored. " \
                            "\n\nThis value is equal to `IntensWidth` * `IntensHeight` * `IntensNumBuckets` * 2 "
__pdoc__[
    'Header.PhaseOriginX'] = "(int)   The X component of the origin of the phase data matrix. " \
                             "\n\nThe coordinate system is the camera’s and (0,0) is in the upper left corner of " \
                             "the video monitor."
__pdoc__[
    'Header.PhaseOriginY'] = "(short)   The Y component of the origin of the phase data matrix. " \
                             "\n\nThe coordinate system is the camera’s and (0,0) is in the upper " \
                             "left corner of the video monitor"
__pdoc__[
    'Header.PhaseWidth'] = "(short)   The width (number of columns) of the phase data matrix. " \
                           "\n\nIf no phase data matrix is present, then this value is zero"
__pdoc__[
    'Header.PhaseHeight'] = "(short)   The height (number of rows) of the phase data matrix. " \
                            "\n\nIf no phase data matrix is present, then this value is zero."
__pdoc__[
    'Header.PhaseNumBytes'] = "(int)   Number of bytes of phase data stored. " \
                              "\n\nThis value is equal to `PhaseWidth` * `PhaseHeight` * 4"
__pdoc__['Header.BEattributes'] = "(`ZygoBEattributes`)  The attributes defined in the `ZygoBEattributes` structure."
__pdoc__['Header.LEattributes'] = "(`ZygoLEattributes`)  The attributes defined in the `ZygoLEattributes` structure."


class DatFile(object):
    """  This class wraps the structure of a Zygo data file stored in the .dat format of MX

    The `openFile` function only loads the measurement metadata from the file header into the object
    The phase data can be loaded in a user array by a subsequent call to `getPhase`
    """

    def __init__(self):
        self.filepath = ""
        """ (`string`) The full path to access the file"""
        self.header = Header()
        """ (`Header`) File header containing all metadata relative to a measurement"""

    def openFile(self, filename=''):
        """         Open a Zygo .dat file in read mode and load the data header in the `self.header` member

            The phase data are not loaded, and can be requested at any time by calling  the `getPhase` method

            Parameters
            ---------------
            `filename` [optional] : the full qualified path of the file to open.\n
            if not given the user will be prompted a File-Open dialog\n
            Returns
            --------------
            **True** if the file was opened and the `header` was succesfully read; **False** otherwise
        """
        if Path(filename).is_file():
            self.filepath = filename
        else:
            tkinter.Tk().withdraw()
            self.filepath = filedialog.askopenfilename(title="open a Zygo.dat file", filetypes=(("Zygo dat", "*.dat"),))
        logger.info(f"Opening {self.filepath}")

        with open(self.filepath, 'rb') as filein:
            filein.readinto(self.header)
        # check if file is valid
        if self.header.MagicNumber - self.header.HeaderFormat != 0x881b036e:
            logger.critical("File is invalid")
            return False
        return True

    def getHeight(self, multiplier=1.):
        """ Reads the phase data from thefile and returns it in a `numpy.array` \n
            Possible invalid data are replaced by NaN in the returned array\n
            Parameters:
            -----------
            `multiplier`: (**float**)  [optional] a multiplicative factor which will be applied to all read values.\n
            The default value is 1., and height values will be expressed in meters.\n
            Returns:
            -------------\n
            (`numpy.array`) The array of read values (in meters) , with invalid data replaced by NaNs, optionnally multiplied by `multiplier`.
            """
        h = self.header
        if h.PhaseNumBytes == 0:
            logger.critical("There is no phase image in this file")
            return np.empty(0)

        phaseOffset = h.HeaderSize + h.IntensNumBytes
        if h.BEattributes.PhaseRes == 0:
            nres = 4096
        elif h.BEattributes.PhaseRes == 1:
            nres = 32768
        else:
            nres = 131072
        # print("Fringe resolution=",nres)
        scale = multiplier * h.BEattributes.IntfScaleFactor * h.BEattributes.ObliquityFactor * \
                h.BEattributes.WavelengthIn / nres

        Phase = np.ndarray(shape=(h.PhaseHeight, h.PhaseWidth))
        linPhase = Phase.ravel()
        with open(self.filepath, 'rb') as file:
            file.seek(phaseOffset)
            i = 0
            for ival in struct.iter_unpack('>l', file.read(h.PhaseNumBytes)):
                if ival[0] >= 2147483640:
                    linPhase[i] = np.nan  # nan can be later used to generate a mask
                else:
                    linPhase[i] = scale * ival[0]
                i += 1
        return Phase

    # return pixel size in m
    def getPixel(self):
        """
        Return
        ------
        the pixel size in  \\(m^{-1}\\) (as a float)
        """

        return self.header.BEattributes.LateralResolution

    def getOrigin(self):
        """
        Return
        ------
        X_origin, Y_origin : (int, in)
            the oordinates (in pixels) of the upper left corner of the phase array in the interferometer camera frame
        """
        return [self.header.PhaseOriginX, self.header.PhaseOriginY]

    def info(self):
        """ Print some information on this file on std_out
        """
        h = self.header
        logger.info("magic num " + str(hex(h.MagicNumber)))
        logger.info("Header format " + str(h.HeaderFormat))
        logger.info("Header size " + str(h.HeaderSize))
        logger.info("User comment " + str(h.BEattributes.Comment))
        logger.info("Software " + str(ZygoSoftwareType[h.SoftInfo.Type]) +
                    " version " + str(h.SoftInfo.Version) +
                    " date " + str(h.SoftInfo.Date) + 'ASCII')
        logger.info("Origin X " + str(self.header.PhaseOriginX) + "  Origin Y " + str(self.header.PhaseOriginY))
        logger.info(f"Data size  {datfile.header.PhaseHeight} x {datfile.header.PhaseWidth}")
        logger.info("Pixel size " + str(self.getPixel()))
        location = self.header.LEattributes.Coordinates
        logger.info(f"Data Location  ({location.XPosition}, {location.YPosition}, {location.ZPosition} )")


if __name__ == "__main__":
    from plotly.express import imshow
    import plotly.io as pio
    import numpy as np
    import xarray as xr
    import sys
    pio.renderers.default = "browser"

    root = logging.getLogger()
    root.setLevel(logging.DEBUG)
    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(logging.DEBUG)
    root.addHandler(handler)
    datfile = DatFile()
    datfile.openFile(r"Y:\Mesures-2024\DI-61020_SWING\A Stitched Scans\Stitched_Scan_207_topStripe.dat")
    datfile.info()
    height = xr.DataArray(datfile.getHeight(),
                          [("x", datfile.getPixel()*(np.arange(datfile.header.PhaseHeight)-datfile.header.PhaseHeight/2)),
                           ("y", datfile.getPixel()*(np.arange(datfile.header.PhaseWidth)-datfile.header.PhaseWidth/2))])
    logger.info(str(height))
    fig = imshow(height)
    fig.show()

