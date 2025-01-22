# -*- coding: utf-8 -*-
"""
Module for loading carpem files.
Allows for easy chages of the file for batch of simulations
Warning : running Carpem only work within the msys64 console used form the carpem environnement 
"""
import re
import numpy as np 
import subprocess
import os

work_dir = r"C:/msys64/home/DENNETIERE/.carpem"
datapath = r"D:/Dennetiere/Programmes_C++/carpem_sources/data"
install_dir = r"D:/Dennetiere/Programmes_C++/carpem/bin"

class InputFile(object):
    def __init__(self, filename=None):
        self.input_data = {}
        self.result_run = {}
        self.result_str = {}
        self.legend_result = []
        self.temporeneo_filename = filename
        self.generated_temporeneo_filename = None
        # Pattern definitions for specific sections
        self.patterns = {
            "version": r"version ([\d.]+)",
            "mca_params": r"\s([\d.]+)\s([\d.]+)\s([\d.]+)\s+\* params MCA : depth dutyC nbr of periods",
            "index_data_type": r"\s(\w+)\s(\d+)\s+\*\s+index data type.*",
            "layer_data": r"\s([\d.]+)\s([\d.]+)\s(\S+)\s\* thickness material",
            "sub_gratings": r"(\d+)\s+\*\s+Number of sub-gratings",
            "henke_path": r"(\S+)\s+\*\s+Path to HENKE data",
            "henke_extension": r"(\.\w+)\s+\*\s+Henke data file extension",
            "orders_computations": r"(\d+)\s+\*\s+Number of orders in computations",
            "substrate": r"(\w+)\s+([\d.]+)\s(\S+)\s+\*\s+SUBSTRATE material",
            "grating_period": r"([\d.]+)\s+\*\s+GRATING PERIOd",
            "layers_in_one_period": r"(\d+)\s+\*\s+Number of layers in one period of grating",
            "period_in_subgrating": r"(\d+)\s+\* Number of layer periods in grating",
            "material_data": r"\s+([\d.]+)\s(\S+)\s\*\s+material in the '(a|b)' area",
            "thickness": r"([\d.]+)\s+\*\s+thickness \(Angström\)",
            "boundaries": r"([\d.]+)\s([\d.]+)\s+\*\s+boundaries of the 'a' area",
            "energy": r"([\d.]+)\s+\*\s+Energy",
            "energy_step": r"([\d.]+)\s+\*\s+energy step",
            "angle": r"([\d.]+)\s+\*\s+angle  or",
            "angle_step": r"([\d.]+)\s+\*\s+angle step",
            "steps": r"([\d.]+)\s+\*\s+number of computation steps",
            "number_windows": r"(\d+)\s+\*\snum windows",
            "order_use": r"(\d+)\s+\*\sOrder of use",
            "intensity_efficiency": r"(\d+)\s+\*\scompute efficiency",
            "exit_angles": r"(\d+)\s+\*\s1 outputs exit angles",
            "polarization": r"(\d+)\s+\*Polarization",
            "n_harm": r"(\d+)\s+\*\snum of computed  harmonics",
            "scan_type": r"(\d+)\s+\*scan type",
            "output_phase": r"(\d+)\s+\*\s1 output phase",
            "regulating_factor": r"(\d+)\s+\*\s+Factor which regulates step",
            "debye_waller": r"([\d\.]+)\s+\*\sDebye-Waller coefficient",
            "vaccum_index": r"([\d\.]+)\s+([\d\.]+)\s+\*\s+Optical index of vacuum",
        }
        if filename is not None:
            self.parse_temporeneo(filename)
        
    def use_default_temporeneo(self):
        filename = os.path.join(work_dir, "temporeneo")
        assert os.path.exists(filename), ValueError(f"No temporeneo file in working directory {work_dir}")
        self.temporeneo_filename = filename
        return filename
        
    def parse_temporeneo(self, filename=None):
        if filename is None :
            filename = self.use_default_temporeneo()
        data = {}
        
        # Process line by line and populate the dictionary
        with open(filename, 'r') as filin:
            for line in filin.readlines():
                for key, pattern in self.patterns.items():
                    match = re.search(pattern, line)
                    if match:
                        if key not in data:
                            data[key] = []
                        data[key].append(match.groups() if len(match.groups()) > 1 else match.group(1))
        
        # Post-processing to clean up single-item lists
        for key, value in data.items():
            if len(value) == 1 and key!='boundaries':
                data[key] = value[0]
        
        self.input_data = data
        return filename
        
    def show_data(self):
        for key, val in self.input_data.items():
            print(key, val)
            
    def change_parameter(self, name, val):
        assert name in self.patterns.keys(), ValueError(f"name must be in {self.patterns.keys()}")
        assert name in self.input_data.keys(), ValueError(f"name must be in {self.input_data.keys()}")
        self.input_data[name] = val
                
    def generate_temporeneo(self, filename=None):
        if filename is None:
            filename = self.use_default_temporeneo()
        self.generated_temporeneo_filename = filename
            
        result = []
         # Header
        result.append(f"version {self.input_data['version']}   * CARPEM version used for building this file *")

        # MCA data section
        if 'mca_params' in self.input_data.keys():
            result.append("# MCA data section")
            result.append(f"# {self.input_data['mca_params'][0]} {self.input_data['mca_params'][1]} {self.input_data['mca_params'][2]}    * params MCA : depth dutyC nbr of periods  *")
            result.append(f"# {self.input_data['index_data_type'][0]} {self.input_data['index_data_type'][1]}     *\t index data type (Henke /Palik) ; nbr of layers / period *")

            # Layer data
            for layer in self.input_data['layer_data']:
                result.append(f"# {layer[0]} {layer[1]} {layer[2]} * thickness material *")

        # Sub-gratings
        result.append(f"{self.input_data['sub_gratings']}  * Number of sub-gratings  *   ")
        result.append(f"{self.input_data['henke_path']}  * Path to HENKE data *")
        result.append(f"{self.input_data['henke_extension']}   * Henke data file extension *")
        result.append(f"{self.input_data['orders_computations']}      * Number of orders in computations:  NORDRES  ( 2(NORDRES+1) waves propagated )  * ")

        # Substrate
        substrate = self.input_data['substrate']
        result.append(f"{substrate[0]}  {substrate[1]} {substrate[2]} * SUBSTRATE material, lowercase  *")

        # Grating period
        result.append(f"{self.input_data['grating_period']}    * GRATING PERIOd, in Angström *   ")

        # Layers in period
        thickness_idx = 0
        for i, layers in enumerate(self.input_data['layers_in_one_period'], start=1):
            result.append(f"{layers}      * Number of layers in one period of grating {i} *     ")

            # Iterate over the layers
            for _ in range(int(layers)):
                result.append("palik  * Using index tables rather than Henke element data *")
                # Material 'a'
                material_a = self.input_data['material_data'][thickness_idx]
                result.append(f" {material_a[0]} {material_a[1]} * material in the '{material_a[2]}' area *")
                # Material 'b'
                material_b = self.input_data['material_data'][thickness_idx + 1]
                result.append(f" {material_b[0]} {material_b[1]} * material in the '{material_b[2]}' area *")
                # Thickness
                result.append(f" {self.input_data['thickness'][thickness_idx // 2]}  * thickness (Angström) * ")
                thickness_idx += 2

            # Boundaries
            result.append(f" {self.input_data['boundaries'][i-1][0]} {self.input_data['boundaries'][i-1][1]}   * boundaries of the 'a' area * ")
            # Number of layer periods
            result.append(f"{self.input_data['period_in_subgrating'][i-1]}  * Number of layer periods in grating {i} * ")
            result.append("0   * variation of period number at each computation step (usually=0) *   ")

        # Energy and angle data
        result.append(f" {self.input_data['energy']}  * Energy * ")
        result.append(f" {self.input_data['angle']}   * angle  or C_RATIO according to scan parameter* ")
        result.append(f" {self.input_data['energy_step']}  * energy step * ")
        result.append(f" {self.input_data['angle_step']}  * angle step * ")
        result.append(f" {self.input_data['steps']}   * number of computation steps * ")
        result.append(f" {self.input_data['number_windows']}   * num windows (+/- displayed orders) *  ")
        result.append(f" {self.input_data['order_use']}        * Order of use (+ is grazing on exit) *   ")
        result.append(f" {self.input_data['intensity_efficiency']}  * compute efficiency (1) or intensity (0) *")
        result.append(f" {self.input_data['exit_angles']}    * 1 outputs exit angles (1) or not (0) * ")
        result.append(f" {self.input_data['polarization']}  *Polarization: 1=S; 2=P * ")
        result.append(f" {self.input_data['n_harm']} * num of computed  harmonics * ")
        result.append(f" {self.input_data['scan_type']}   *scan type 0= fixed incidence; 1=fixed deviation; 3=fixed Omega; 4 fixed C ratio *")
        result.append(f" {self.input_data['output_phase']}   * 1 output phase, 0 no phase output *")
        result.append(f" {self.input_data['regulating_factor']} *  Factor which regulates step refining in RK integration: 8 is safe *")
        result.append(f" {self.input_data['debye_waller']} * Debye-Waller coefficient of line placement, if 0 no phase uncertainty * ")
        result.append(f" {self.input_data['vaccum_index'][0]} {self.input_data['vaccum_index'][1]}  *  Optical index of vacuum (may be changed if external material is not vacuum) * ")
        
        
        with open(filename, "w") as filout:
            filout.writelines("\n".join(result))
            
        return filename

        
    def run(self, filename=None):
        filename = self.parse_temporeneo(filename)
        filename = self.generate_temporeneo(filename)
        ret = subprocess.run(r"RR2 "+filename, shell=True, capture_output=True)
        self.return_str = ret.stdout.decode('latin-1')
        self.parse_return_run()
        assert self.result_run["harmonic 1"].shape[0] == int(self.input_data["steps"]), \
            RuntimeError(f"Something went wrong during run, RR2 returned : {ret.stderr.decode('latin-1')}")
        
    def parse_return_run(self):
        self.result_run = {}
        for i in range(1, int(self.input_data['n_harm'])+1):
            self.result_run[f"harmonic {i}"] = np.loadtxt(os.path.join(work_dir,f"har{i}"))
        with open(os.path.join(work_dir,"har1"), "r") as filin:
            for line in filin.readlines():
                if line[0] == "#":
                    self.legend_result = [line[i: i+14].replace("#",'').strip() for i in range(0, len(line)-1, 14)]
            
class CarpemFile(InputFile):
    def __init__(self, filename):
        super().__init__(filename)
        self.output_data = None
        self.data_filename = filename
        
    def read_data(self):
        with open(self.data_filename, "r") as filin:
            for line in filin.readlines():
                if line[0] == "#":
                    self.legend_result = [line[i: i+14].replace("#",'').strip() for i in range(0, len(line)-1, 14)]
        self.output_data = np.loadtxt(self.data_filename)
        
if __name__ == "__main__":
    fi = InputFile()
    fi.parse_temporeneo()
    fi.show_data()
    #print(fi.input_data)
    #print(fi.generate_temporeneo("temporeneo_regen"))
    fi.change_parameter("n_harm", 2)
    new_temporeneo = fi.generate_temporeneo(os.path.join(work_dir,"temporeneo_regen"))
    fi.run(new_temporeneo)
    print("return:")
    print(fi.legend_result)
    print(fi.result_run)
    print(fi.result_run["harmonic 1"].shape)
    print()

    cf = CarpemFile(r"D:\Dennetiere\Programmes_C++\carpem\data\C20-Pt-d100-S-H-2-0,98") 
    cf.show_data()
    cf.read_data()
    print(cf.output_data)
    new_temporeneo_cf = cf.generate_temporeneo(os.path.join(work_dir,"temporeneo_from_carpemfile"))
    cf.run(new_temporeneo_cf)
    print(cf.result_run)
    