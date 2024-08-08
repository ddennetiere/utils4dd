import os
import re
import xml.etree.ElementTree as ET


class LogBook(object):
    def __init__(self, filename):
        self.filename = filename
        self.xml = None
        self.data_dir = None
        self.filepattern = None
        self.scans = []
        self.update()

    def update(self):
        self.scans = []
        self.xml = ET.parse(self.filename)
        self.data_dir = self.xml.find("datadir").text.strip()
        self.filepattern = self.xml.find("filepattern").text.strip()
        for scan in self.xml.findall("scan"):
            self.scans.append(scan.get("name"))

    def get_scan_metadata(self, scan_number, metadata, dtype=str):
        self.update()
        for scan in self.xml.findall("scan"):
            if scan.get("name") == str(scan_number):
                try:
                    return dtype(scan.find(metadata).text.strip())
                except ValueError:
                    return scan.find(metadata).text.strip()
        raise AttributeError(f"Scan not found")

    def get_scan_filename(self, scan_number):
        self.update()
        metadata_dict = {"[yyyy]": "date0", "[MM]": "date1", "[dd]": "date2"}
        subs_keys = re.findall("\[[a-zA-Z]+\]", self.filepattern)
        for key in set(subs_keys):
            assert key in metadata_dict.keys()
            if "date" in metadata_dict[key]:
                date = self.get_scan_metadata(scan_number, "date").strip().split("-")
                metadata = date[int(metadata_dict[key][-1])]
            else:
                metadata = self.get_scan_metadata(scan_number, metadata_dict[key])
            metadata_dict[key] = metadata
        fileindex = re.findall("\[\d+i\]", self.filepattern)
        metadata_dict[fileindex[0]] = str(scan_number).zfill(int(fileindex[0][1]))
        return os.path.join(self.data_dir, fill_pattern(self.filepattern, metadata_dict))


def fill_pattern(pattern, pattern_vars):
    filename = pattern
    for key in pattern_vars.keys():
        filename = filename.replace(key, pattern_vars[key])
    return filename


if __name__ == "__main__":
    logtest = LogBook(r'D:\Dennetiere\Upgrade\Onduleur bipériodique\logbook_20240402.xml')
    print(logtest.get_scan_metadata(63, "description"))
    print(logtest.get_scan_filename(63))
