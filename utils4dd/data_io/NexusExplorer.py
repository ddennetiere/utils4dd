import re
import sys
import h5py
import numpy as np
import csv
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                             QHBoxLayout, QTreeWidget, QTreeWidgetItem, QPushButton,
                             QFileDialog, QSplitter, QLabel, QMessageBox, 
                              QFrame, QDialog, QComboBox, QDialogButtonBox)


from PyQt5.QtCore import Qt, pyqtSignal, QMimeData,  QByteArray
from PyQt5.QtGui import QFont, QDrag, QPixmap, QImage
from PyQt5.QtSvg import QSvgWidget
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure
import matplotlib
import traceback
matplotlib.use('Qt5Agg')


class DataViewer(QDialog):
    def __init__(self, data, name, parent=None):
        super().__init__(parent)
        self.data = data
        self.name = name
        self.setWindowTitle(f"Data Viewer - {name}")
        self.resize(600, 200)

        self.layout = QVBoxLayout(self)
        self.display_widget = None

        self._build_ui()

    def _is_svg(self, text):
        """Détermine si le texte correspond à du contenu SVG."""
        if not isinstance(text, str):
            return False
        lower = text.strip().lower()
        return lower.startswith("<svg") or (lower.startswith("<?xml") and "<svg" in lower)

    def _convert_data_to_str(self, data):
        """Convertit les données en texte lisible."""
        unit = data.attrs.get('units')
        if unit is None: 
            unit = ""
        unit = " " + unit

        if isinstance(data[()], bytes):
            try:
                return data[()].decode("utf-8"), "string"
            except UnicodeDecodeError:
                return repr(data[()]), "string"
        elif isinstance(data[()], (str, np.str_)):
            return str(data[()]) , "string"
        elif np.isscalar(data[()]):
            return str(data[()]) + unit, "string"
        elif isinstance(data[()], np.ndarray):
            return data[()], "image"
        else:
            return repr(data) + f"\n\nValue : {str(np.array(data))}{unit}", "string"

    def _build_ui(self):
        data_str, data_type = self._convert_data_to_str(self.data)

        if data_type =="string":
            if self._is_svg(data_str):
                self.resize(1000, 800)
                svg_widget = QSvgWidget()
                svg_widget.load(QByteArray(data_str.encode("utf-8")))
                svg_widget.setMinimumSize(800, 600)
                self.display_widget = svg_widget
                self.layout.addWidget(svg_widget)
            else:
                # Affichage direct de la valeur (string ou scalaire)
                label = QLabel(data_str)
                label.setTextInteractionFlags(Qt.TextSelectableByMouse)
                label.setWordWrap(True)
                self.display_widget = label
                self.layout.addWidget(label)
        elif data_type == "image":
            qImg = QPixmap(QImage(data_str.data, data_str.shape[0], data_str.shape[1], QImage.Format_RGBA8888))
            label = QLabel(self, alignment=Qt.AlignCenter)
            label.setPixmap(qImg)
            self.display_widget = label
            self.layout.addWidget(label)

    def show(self):
        super().show()



class DropZone(QFrame):
    """Zone de dépôt pour le drag & drop"""
    
    dataset_dropped = pyqtSignal(object, str, str)  # dataset, name, zone_type
    
    def __init__(self, zone_type, parent=None):
        super().__init__(parent)
        self.zone_type = zone_type  # 'x_axis', 'overlay', 'secondary_y', 'secondary_x'
        self.setAcceptDrops(True)
        self.is_highlight = False
        self.setup_ui()
        
    def setup_ui(self):
        """Configure l'apparence de la zone de dépôt"""
        self.setFrameStyle(QFrame.StyledPanel)
        self.setLineWidth(2)
        self.setStyleSheet("""
            QFrame {
                background-color: rgba(200, 200, 200, 0.3);
                border: 2px dashed #999;
                border-radius: 8px;
            }
        """)
        
        # Texte explicatif selon le type de zone
        layout = QVBoxLayout(self)
        label = QLabel()
        label.setAlignment(Qt.AlignCenter)
        label.setWordWrap(True)
        
        if self.zone_type == 'x_axis':
            label.setText("ZONE X1\nGlissez ici pour\ndéfinir l'axe X")
            label.setStyleSheet("color: #2196F3; font-weight: bold;")
        elif self.zone_type == 'overlay':
            label.setText("ZONE Y1\nGlissez ici pour\nsuperposer")
            label.setStyleSheet("color: #FF9800; font-weight: bold;")
        elif self.zone_type == 'secondary_y':
            label.setText("ZONE Y2\nGlissez ici pour\naxe Y secondaire")
            label.setStyleSheet("color: #E91E63; font-weight: bold;")
        elif self.zone_type == 'secondary_x':
            label.setText("ZONE X2\nGlissez ici pour\naxe X secondaire")
            label.setStyleSheet("color: #35943F; font-weight: bold;")
            
        layout.addWidget(label)
        
    def dragEnterEvent(self, event):
        """Gère l'entrée d'un élément glissé"""
        if event.mimeData().hasText():
            event.acceptProposedAction()
            self.highlight(True)
            
    def dragLeaveEvent(self, event):
        """Gère la sortie d'un élément glissé"""
        self.highlight(False)
        
    def dropEvent(self, event):
        """Gère le dépôt d'un élément"""
        self.highlight(False)
        
        if event.mimeData().hasText():
            # Récupère les données du dataset depuis le mime data
            data_str = event.mimeData().text()
            try:
                # Format: "dataset_path|dataset_name"
                dataset_path, dataset_name = data_str.split('|', 1)
                
                # Récupère le dataset depuis le fichier HDF5 ouvert
                parent_widget = self.parent()
                while parent_widget and not hasattr(parent_widget, 'tree_widget'):
                    parent_widget = parent_widget.parent()
                    
                if parent_widget and parent_widget.tree_widget.hdf5_file:
                    dataset = parent_widget.tree_widget.hdf5_file[dataset_path]
                    self.dataset_dropped.emit(dataset, dataset_name, self.zone_type)
                    event.acceptProposedAction()
                    
            except Exception as e:
                print(f"Erreur lors du drop: {e}")
                
    def highlight(self, enabled):
        """Met en surbrillance la zone de dépôt"""
        self.is_highlight = enabled
        if enabled:
            color = "#4CAF50" if self.zone_type == 'x_axis' else "#FF5722" if self.zone_type == 'overlay' else "#9C27B0"
            self.setStyleSheet(f"""
                QFrame {{
                    background-color: rgba(76, 175, 80, 0.2);
                    border: 3px solid {color};
                    border-radius: 8px;
                }}
            """)
        else:
            self.setStyleSheet("""
                QFrame {
                    background-color: rgba(200, 200, 200, 0.3);
                    border: 2px dashed #999;
                    border-radius: 8px;
                }
            """)


class PlotCanvas(FigureCanvas):
    """Widget pour afficher les graphiques matplotlib avec support multi-axes"""
    
    def __init__(self, parent=None, width=8, height=6, dpi=100):
        self.fig = Figure(figsize=(width, height), dpi=dpi)
        super().__init__(self.fig)
        self.setParent(parent)
        self.axes = self.fig.add_subplot(111)
        self.axes2 = None  # Axe Y secondaire
        self._img_colorbar = None
        
        # Stockage des datasets
        self.datasets = {
            'main': None,
            'x_axis': None,
            'secondary_x': None,
            'overlay': [],
            'secondary_y': []
        }
        self.dataset_names = {
            'main': '',
            'x_axis': '',
            'secondary_x': '',
            'overlay': [],
            'secondary_y': []
        }
        
    def clear_plot(self):
        """Efface le graphique et remet à zéro les datasets"""
        self.axes.clear()
        # remove image colorbar if present
        if getattr(self, '_img_colorbar', None) is not None:
            try:
                cb = self._img_colorbar
                cb.remove()
                # also remove the colorbar axes from the figure so layout is restored
                try:
                    if hasattr(cb, 'ax') and cb.ax in self.fig.axes:
                                self.fig.delaxes(cb.ax)
                                try:
                                    self._using_custom_cax = False
                                except Exception:
                                    pass
                except Exception:
                    pass
            except Exception:
                pass
            self._img_colorbar = None
            # restore previous axes position if we saved it
            try:
                if getattr(self, '_axes_pos_before_colorbar', None) is not None:
                    try:
                        self.axes.set_position(self._axes_pos_before_colorbar)
                    except Exception:
                        pass
                    try:
                        pos = self.axes.get_position()
                        if pos.width < 0.5:
                            self.axes.set_position([0.125, 0.11, 0.775, 0.77])
                    except Exception:
                        pass
                    try:
                        del self._axes_pos_before_colorbar
                    except Exception:
                        pass
            except Exception:
                pass
        if self.axes2:
            self.axes2.remove()
            self.axes2 = None
        self.datasets = {
            'main': None,
            'x_axis': None,
            'secondary_x': None,
            'overlay': [],
            'secondary_y': []
        }
        self.dataset_names = {
            'main': '',
            'x_axis': '',
            'secondary_x': '',
            'overlay': [],
            'secondary_y': []
        }
        self.draw()
        
    def set_main_dataset(self, data, name):
        """Définit le dataset principal"""
        self.datasets['main'] = data
        self.dataset_names['main'] = name
        self.update_plot()
        
    def add_x_axis_dataset(self, data, name):
        """Définit le dataset pour l'axe X"""
        self.datasets['x_axis'] = data
        self.dataset_names['x_axis'] = name
        self.update_plot()
        
    def add_overlay_dataset(self, data, name):
        """Ajoute un dataset en superposition"""
        self.datasets['overlay'].append(data)
        self.dataset_names['overlay'].append(name)
        self.update_plot()
        
    def add_secondary_y_dataset(self, data, name):
        """Ajoute un dataset sur l'axe Y secondaire"""
        self.datasets['secondary_y'].append(data)
        self.dataset_names['secondary_y'].append(name)
        self.update_plot()

    def add_secondary_x_dataset(self, data, name):
        """Ajoute un dataset sur l'axe X secondaire"""
        self.datasets['secondary_x'] = data
        self.dataset_names['secondary_x'] = name
        self.update_plot()
        
    def update_plot(self):
        """Met à jour le graphique avec tous les datasets"""
        self.axes.clear()
        if self.axes2:
            self.axes2.clear()
            
        main_data = self.datasets['main']
        x_data = self.datasets['x_axis']
        
        if main_data is None:
            return
            
        try:
            # Prépare les données X
            if x_data is not None and x_data.ndim == 1:
                x_values = x_data
                x_label = self.dataset_names['x_axis']
            else:
                x_values = np.arange(len(main_data.flatten()) if main_data.ndim > 1 else len(main_data))
                x_label = "Index"
                
            # Dataset principal
            # If the main data is 2D, show it as an image instead of flattening
            if getattr(main_data, 'ndim', 1) == 2:
                img = main_data
                # Determine extent from provided axes if available
                extent = None
                # If x axis corresponds to columns
                if x_data is not None and getattr(x_data, 'ndim', 1) == 1 and len(x_data) == img.shape[1]:
                    x0, x1 = float(x_data[0]), float(x_data[-1])
                else:
                    x0, x1 = 0, img.shape[1] - 1

                # try to find a y axis from the secondary_x dataset (commonly used for rows)
                y_vals = None
                sec_x = self.datasets.get('secondary_x')
                if sec_x is not None and getattr(sec_x, 'ndim', 1) == 1 and len(sec_x) == img.shape[0]:
                    y_vals = sec_x
                    y0, y1 = float(y_vals[0]), float(y_vals[-1])
                else:
                    y0, y1 = 0, img.shape[0] - 1

                extent = [x0, x1, y0, y1]

                im = self.axes.imshow(img, origin='lower', aspect='auto', extent=extent, cmap='viridis')
                # remove previous colorbar if any to avoid piling up
                if getattr(self, '_img_colorbar', None) is not None:
                    try:
                        cb = self._img_colorbar
                        cb.remove()
                        try:
                            if hasattr(cb, 'ax') and cb.ax in self.fig.axes:
                                self.fig.delaxes(cb.ax)
                        except Exception:
                            pass
                    except Exception:
                        pass
                    self._img_colorbar = None
                # Create a dedicated colorbar axes next to the main axes and shrink
                # the main axes slightly so the colorbar doesn't overlay it. Save
                # the original axes position so it can be restored on removal.
                try:
                    main_pos = self.axes.get_position()
                    # pad between main axes and colorbar (figure fraction)
                    pad = 0.02
                    # colorbar axes width (figure fraction)
                    cax_width = 0.04
                    # extra horizontal room to account for tick labels
                    label_pad = 0.04
                    # Save original position once
                    if not hasattr(self, '_axes_pos_before_colorbar'):
                        try:
                            self._axes_pos_before_colorbar = main_pos
                        except Exception:
                            self._axes_pos_before_colorbar = None

                    # calculate new main width and cax position
                    new_main_width = main_pos.width - (cax_width + pad + label_pad)
                    if new_main_width < 0.2:
                        # don't shrink too much; reduce by 10% instead
                        new_main_width = max(main_pos.width * 0.9, 0.2)
                        cax_width = main_pos.width - new_main_width - pad - label_pad
                        if cax_width <= 0:
                            cax_width = 0.03

                    # slightly inset vertically so colorbar ticks/labels fit
                    vpad = main_pos.height * 0.02
                    new_main_pos = [main_pos.x0, main_pos.y0, new_main_width, main_pos.height]
                    try:
                        self.axes.set_position(new_main_pos)
                    except Exception:
                        pass

                    cax_x = main_pos.x0 + new_main_width + pad
                    cax_rect = [cax_x, main_pos.y0 + vpad, cax_width, max(main_pos.height - 2 * vpad, 0.05)]
                    cax = self.fig.add_axes(cax_rect)
                    # Create the colorbar in the dedicated axes and ensure ticks/labels
                    # are drawn on the right side and not clipped.
                    self._img_colorbar = self.fig.colorbar(im, cax=cax)
                    try:
                        cax.tick_params(axis='y', which='major', pad=3)
                    except Exception:
                        pass
                    self._using_custom_cax = True
                except Exception:
                    # fallback to the default behavior if custom axes fail
                    try:
                        self._img_colorbar = self.fig.colorbar(im, ax=self.axes, fraction=0.046, pad=0.04)
                        self._using_custom_cax = False
                    except Exception:
                        self._img_colorbar = None
                        self._using_custom_cax = False
                self.axes.set_xlabel(self.dataset_names.get('x_axis', ''))
                self.axes.set_ylabel(self.dataset_names.get('main', ''))
                self.axes.set_title(self.dataset_names.get('main', ''))
                # For image display we skip overlays and secondary y plotting
            else:
                # ensure any previous image colorbar is removed when switching to line plot
                if getattr(self, '_img_colorbar', None) is not None:
                    try:
                        cb = self._img_colorbar
                        cb.remove()
                        try:
                            if hasattr(cb, 'ax') and cb.ax in self.fig.axes:
                                # delete the colorbar axes
                                self.fig.delaxes(cb.ax)
                                # mark that we no longer use a custom cax
                                self._using_custom_cax = False
                        except Exception:
                            pass
                    except Exception:
                        pass
                    self._img_colorbar = None
                    # restore layout so the plot area expands back
                    try:
                        # if main axes was removed for some reason, recreate it
                        if self.axes not in self.fig.axes:
                            self.axes = self.fig.add_subplot(111)
                        # restore previous axes position if we saved it
                        if getattr(self, '_axes_pos_before_colorbar', None) is not None:
                            try:
                                self.axes.set_position(self._axes_pos_before_colorbar)
                            except Exception:
                                pass
                            try:
                                # if the restored width is still too small, force a sensible default
                                pos = self.axes.get_position()
                                if pos.width < 0.5:
                                    self.axes.set_position([0.125, 0.11, 0.775, 0.77])
                            except Exception:
                                pass
                            try:
                                del self._axes_pos_before_colorbar
                            except Exception:
                                pass
                        # run tight_layout once to normalize padding
                        try:
                            self.fig.tight_layout()
                        except Exception:
                            pass
                    except Exception:
                        pass
                main_y = main_data.flatten() if main_data.ndim > 1 else main_data
                if len(x_values) != len(main_y):
                    # Ajuste la longueur si nécessaire
                    min_len = min(len(x_values), len(main_y))
                    x_values = x_values[:min_len]
                    main_y = main_y[:min_len]
                    
                _ = self.axes.plot(x_values, main_y, 'b-', linewidth=2, 
                                      label=self.dataset_names['main'])
                self.axes.set_ylabel(self.dataset_names['main'], color='b')
                self.axes.tick_params(axis='y', labelcolor='b')
            
            # Datasets en superposition
            colors = ['green', 'red', 'purple', 'orange', 'brown']
            for i, (overlay_data, overlay_name) in enumerate(zip(self.datasets['overlay'], 
                                                               self.dataset_names['overlay'])):
                overlay_y = overlay_data.flatten() if overlay_data.ndim > 1 else overlay_data
                if len(x_values) >= len(overlay_y):
                    x_plot = x_values[:len(overlay_y)]
                else:
                    x_plot = x_values
                    overlay_y = overlay_y[:len(x_values)]
                    
                color = colors[i % len(colors)]
                self.axes.plot(x_plot, overlay_y, color=color, linewidth=2, 
                             linestyle='--', label=overlay_name)
                             
            # Datasets sur axe Y secondaire
            if self.datasets['secondary_y']:
                self.axes2 = self.axes.twinx()
                
                for i, (sec_data, sec_name) in enumerate(zip(self.datasets['secondary_y'], 
                                                            self.dataset_names['secondary_y'])):
                    sec_y = sec_data.flatten() if sec_data.ndim > 1 else sec_data
                    if len(x_values) >= len(sec_y):
                        x_plot = x_values[:len(sec_y)]
                    else:
                        x_plot = x_values
                        sec_y = sec_y[:len(x_values)]
                        
                    color = colors[(i + 2) % len(colors)]
                    self.axes2.plot(x_plot, sec_y, color=color, linewidth=2, 
                                   linestyle=':', label=f"{sec_name} (Y2)")
                                   
                self.axes2.set_ylabel(self.dataset_names['secondary_y'], color='r')
                self.axes2.tick_params(axis='y', labelcolor='r')
                
            if self.datasets['secondary_x'] is not None:
                sec_x = self.datasets["secondary_x"]
                try:
                    len_sec = len(sec_x)
                except Exception:
                    len_sec = None
                # only create secondary axis mapping if lengths match
                if len_sec is not None and len(x_values) == len_sec:
                    def forward(x):
                        return np.interp(x, x_values, sec_x)
                    def inverse(x):
                        return np.interp(x, sec_x, x_values)
                    self.axes3 = self.axes.secondary_xaxis("top", functions=(forward, inverse))
                else:
                    # Inform user in the parent UI if available
                    parent = getattr(self, 'parent', None) and self.parent()
                    if parent is not None and hasattr(parent, 'info_label'):
                        try:
                            parent.info_label.setText(f"Axis length mismatch: secondary_x length={len_sec} != x values length={len(x_values)}")
                        except Exception:
                            pass

            self.axes.set_xlabel(x_label)
            self.axes.grid(True, alpha=0.3)
            
            # Légende combinée et limites (non pertinentes pour images)
            if not (getattr(main_data, 'ndim', 1) == 2):
                lines1, labels1 = self.axes.get_legend_handles_labels()
                lines2, labels2 = [], []
                if self.axes2:
                    lines2, labels2 = self.axes2.get_legend_handles_labels()
                if lines1 or lines2:
                    self.axes.legend(lines1 + lines2, labels1 + labels2, 
                                   loc='upper right', bbox_to_anchor=(1, 1))
                self.axes.set_xlim(left=x_values.min(), right=x_values.max())
            else:
                # For images, let matplotlib handle limits/axis automatically
                pass
            # Avoid tight_layout when using custom colorbar axes (incompatible
            # with tight_layout in some Matplotlib versions and causes warnings).
            if not getattr(self, '_using_custom_cax', False):
                try:
                    self.fig.tight_layout()
                except Exception:
                    pass
            self.draw()
            
        except Exception as e:
            self.axes.clear()
            self.axes.text(0.5, 0.5, f"Erreur d'affichage:\n{str(e)} \n\n {traceback.format_exc()}", 
                          transform=self.axes.transAxes, ha='center', va='center')
            self.draw()
            
    def export_to_csv(self, file_path):
        """Exporte toutes les données affichées vers un fichier CSV"""
        try:
            if self.datasets['main'] is None:
                raise ValueError("Aucune donnée à exporter")
                
            # Prépare les données pour l'export
            export_data = {}
            
            # Données X
            main_data = self.datasets['main']
            x_data = self.datasets['x_axis']
            
            if x_data is not None and x_data.ndim == 1:
                x_values = x_data
                x_name = self.dataset_names['x_axis']
            else:
                x_values = np.arange(len(main_data.flatten()) if main_data.ndim > 1 else len(main_data))
                x_name = "Index"
                
            # Dataset principal
            main_y = main_data.flatten() if main_data.ndim > 1 else main_data
            if len(x_values) != len(main_y):
                min_len = min(len(x_values), len(main_y))
                x_values = x_values[:min_len]
                main_y = main_y[:min_len]
                
            # Ajoute les données à exporter
            export_data[x_name] = x_values
            export_data[self.dataset_names['main']] = main_y
            
            # Datasets en superposition
            for i, (overlay_data, overlay_name) in enumerate(zip(self.datasets['overlay'], 
                                                               self.dataset_names['overlay'])):
                overlay_y = overlay_data.flatten() if overlay_data.ndim > 1 else overlay_data
                if len(x_values) >= len(overlay_y):
                    overlay_y_export = overlay_y
                else:
                    overlay_y_export = np.full(len(x_values), np.nan)
                    overlay_y_export[:len(overlay_y)] = overlay_y
                    
                export_data[overlay_name] = overlay_y_export
                
            # Datasets sur axe Y secondaire
            for i, (sec_data, sec_name) in enumerate(zip(self.datasets['secondary_y'], 
                                                        self.dataset_names['secondary_y'])):
                sec_y = sec_data.flatten() if sec_data.ndim > 1 else sec_data
                if len(x_values) >= len(sec_y):
                    sec_y_export = sec_y
                else:
                    sec_y_export = np.full(len(x_values), np.nan)
                    sec_y_export[:len(sec_y)] = sec_y
                    
                export_data[f"{sec_name}_Y2"] = sec_y_export
                
            # Écrit le fichier CSV
            with open(file_path, 'w', newline='', encoding='utf-8') as csvfile:
                if export_data:
                    writer = csv.writer(csvfile)
                    
                    # En-têtes
                    headers = list(export_data.keys())
                    writer.writerow(headers)
                    
                    # Données (transpose pour avoir les colonnes)
                    max_len = max(len(data) for data in export_data.values())
                    
                    for i in range(max_len):
                        row = []
                        for header in headers:
                            data = export_data[header]
                            if i < len(data):
                                value = data[i]
                                # Gère les valeurs NaN
                                if np.isnan(value) if isinstance(value, (int, float)) else False:
                                    row.append('')
                                else:
                                    row.append(value)
                            else:
                                row.append('')
                        writer.writerow(row)
                        
            return True, f"Export réussi: {len(export_data)} colonnes, {max_len} lignes"
            
        except Exception as e:
            return False, f"Erreur lors de l'export: {str(e)}"


class AxisSelectionDialog(QDialog):
    """Dialog to choose two axes (X and Y) from a list of candidate axis names."""
    def __init__(self, axis_names, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Sélection des axes pour scatter")
        self.setModal(True)
        self.resize(360, 120)

        self.axis_names = axis_names or []

        layout = QVBoxLayout(self)

        label = QLabel("Sélectionnez l'axe X et l'axe Y (par nom):")
        layout.addWidget(label)

        row = QHBoxLayout()
        self.combo_x = QComboBox()
        self.combo_y = QComboBox()
        self.combo_x.addItems(self.axis_names)
        self.combo_y.addItems(self.axis_names)
        row.addWidget(QLabel("X:"))
        row.addWidget(self.combo_x)
        row.addWidget(QLabel("Y:"))
        row.addWidget(self.combo_y)
        layout.addLayout(row)

        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    def get_selection(self):
        """Return the selected (x_name, y_name) tuple."""
        return (self.combo_x.currentText(), self.combo_y.currentText())


class HDF5TreeWidget(QTreeWidget):
    """Widget personnalisé pour explorer la structure HDF5 avec drag & drop"""
    
    item_selected = pyqtSignal(object, str)
    
    def __init__(self):
        super().__init__()
        self.setHeaderLabel("Structure HDF5")
        self.hdf5_file = None
        self.setDragEnabled(True)
        self.setDragDropMode(QTreeWidget.DragOnly)
        self.itemClicked.connect(self.on_item_clicked)
        
    def load_hdf5_file(self, filepath):
        """Charge un fichier HDF5 et construit l'arbre"""
        try:
            if self.hdf5_file:
                self.hdf5_file.close()
                
            self.hdf5_file = h5py.File(filepath, 'r')
            self.clear()
            
            root_item = QTreeWidgetItem(self)
            root_item.setText(0, filepath.split('/')[-1])
            root_item.setData(0, Qt.UserRole, ('file', self.hdf5_file, ''))
            
            self._add_hdf5_items(self.hdf5_file, root_item, '')
            self.expandAll()
            
        except Exception as e:
            QMessageBox.critical(self, "Erreur", f"Impossible de charger le fichier HDF5:\n{str(e)}")
            
    def _add_hdf5_items(self, hdf5_group, parent_item, parent_path):
        """Ajoute récursivement les éléments HDF5 à l'arbre"""
        for key in hdf5_group.keys():
            item = QTreeWidgetItem(parent_item)
            current_path = f"{parent_path}/{key}" if parent_path else key
            
            obj = hdf5_group[key]
            
            if isinstance(obj, h5py.Group):
                item.setData(0, Qt.UserRole, ('group', obj, current_path))
                item.setText(0, f"📁 {key}")
                self._add_hdf5_items(obj, item, current_path)
            elif isinstance(obj, h5py.Dataset):
                item.setData(0, Qt.UserRole, ('dataset', obj, current_path))
                shape_info = f" {obj.shape}" if obj.shape else ""
                dtype_info = f" ({obj.dtype})"
                item.setText(0, f"📊 {key}{shape_info}{dtype_info}")
                
    def startDrag(self, supportedActions):
        """Démarre le drag pour un dataset"""
        item = self.currentItem()
        if item:
            data = item.data(0, Qt.UserRole)
            if data and data[0] == 'dataset':
                obj_type, dataset, path = data
                
                # Crée le drag
                drag = QDrag(self)
                mimeData = QMimeData()
                mimeData.setText(f"{path}|{item.text(0)[1:]}")
                drag.setMimeData(mimeData)
                
                # Execute le drag
                drag.exec_(Qt.CopyAction)
                
    def on_item_clicked(self, item):
        """Gère le clic sur un élément de l'arbre"""
        data = item.data(0, Qt.UserRole)
        if data:
            obj_type, obj, path = data
            # Emit the object (dataset or group) and the display name; the caller
            # will decide how to handle groups (e.g. NXdata) vs datasets.
            self.item_selected.emit(obj, item.text(0)[1:])
                
    def closeEvent(self, event):
        """Ferme proprement le fichier HDF5"""
        if self.hdf5_file:
            self.hdf5_file.close()
        event.accept()


class HDF5Explorer(QMainWindow):
    """Fenêtre principale de l'explorateur HDF5"""
    
    def __init__(self):
        super().__init__()
        self.init_ui()
        
    def init_ui(self):
        """Initialise l'interface utilisateur"""
        self.setWindowTitle("Explorateur HDF5 - Visualiseur Multi-Axes")
        self.setGeometry(100, 100, 1400, 900)
        
        # Widget central
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        # Layout principal
        main_layout = QHBoxLayout(central_widget)
        
        # Splitter pour diviser l'interface
        splitter = QSplitter(Qt.Horizontal)
        main_layout.addWidget(splitter)
        
        # === BARRE LATÉRALE ===
        sidebar = QWidget()
        sidebar.setMaximumWidth(350)
        sidebar.setMinimumWidth(300)
        sidebar_layout = QVBoxLayout(sidebar)
        
        # Bouton pour ouvrir un fichier
        self.open_button = QPushButton("📁 Ouvrir fichier HDF5")
        self.open_button.clicked.connect(self.open_file)
        sidebar_layout.addWidget(self.open_button)
        
        # Instructions drag & drop
        instructions = QLabel("""
<b>Instructions Drag & Drop:</b><br>
• <span style='color: #2196F3;'>Zone X1</span>: Axe X1 principal (défault: Index)<br>
• <span style='color: #FF9800;'>Zone Y1</span>: Axe Y1 principal<br>
• <span style='color: #E91E63;'>Zone Y2</span>: Axe Y2 secondaire<br>
• <span style='color: #35943F;'>Zone X2</span>: Axe X2 secondaire<br><br>
<i>Glissez les datasets 📊 vers les zones</i>
        """)
        instructions.setWordWrap(True)
        instructions.setStyleSheet("background-color: #e3f2fd; padding: 8px; border-radius: 4px;")
        sidebar_layout.addWidget(instructions)
        
        # Arbre de navigation HDF5
        self.tree_widget = HDF5TreeWidget()
        self.tree_widget.item_selected.connect(self.on_dataset_selected)
        sidebar_layout.addWidget(self.tree_widget)
        
        # Boutons d'actions
        buttons_layout = QHBoxLayout()
        
        # Bouton pour effacer le graphique
        self.clear_button = QPushButton("🗑️ Effacer")
        self.clear_button.clicked.connect(self.clear_plot)
        self.clear_button.setStyleSheet("background-color: #f44336;")
        buttons_layout.addWidget(self.clear_button)
        
        # Bouton pour exporter en CSV
        self.export_button = QPushButton("📁 Export CSV")
        self.export_button.clicked.connect(self.export_to_csv)
        self.export_button.setStyleSheet("background-color: #2196F3;")
        buttons_layout.addWidget(self.export_button)
        
        sidebar_layout.addLayout(buttons_layout)
        
        # Informations sur les datasets
        self.info_label = QLabel("Sélectionnez un dataset pour voir les informations")
        self.info_label.setWordWrap(True)
        self.info_label.setStyleSheet("background-color: #f0f0f0; padding: 10px; border-radius: 5px;")
        sidebar_layout.addWidget(self.info_label)
        
        splitter.addWidget(sidebar)
        
        # === ZONE PRINCIPALE D'AFFICHAGE ===
        plot_container = QWidget()
        plot_container_layout = QVBoxLayout(plot_container)
        
        # Titre de la zone de graphique
        self.plot_title = QLabel("Zone d'affichage des graphiques multi-axes")
        self.plot_title.setAlignment(Qt.AlignCenter)
        self.plot_title.setFont(QFont("Arial", 14, QFont.Bold))
        
        # Layout avec zones de drop
        plot_with_drops = QWidget()
        plot_layout = QVBoxLayout(plot_with_drops)
        plot_layout.setContentsMargins(0, 0, 0, 0)
        
        # Zone de drop supérieure (axe X secondaire)
        self.drop_zone_secondary_x = DropZone('secondary_x')
        self.drop_zone_secondary_x.setMaximumHeight(60)
        self.drop_zone_secondary_x.dataset_dropped.connect(self.on_dataset_dropped)
        
        # Layout horizontal pour le graphique principal avec zones latérales
        middle_layout = QHBoxLayout()
        middle_layout.setContentsMargins(0, 0, 0, 0)
        
        # Zone de drop gauche (superposition)
        self.drop_zone_overlay = DropZone('overlay')
        self.drop_zone_overlay.setMaximumWidth(80)
        self.drop_zone_overlay.dataset_dropped.connect(self.on_dataset_dropped)
        
        # Canvas matplotlib
        self.plot_canvas = PlotCanvas(self, width=10, height=8)
        toolbar = NavigationToolbar(self.plot_canvas, self)
        
        # Zone de drop droite (axe Y secondaire)
        self.drop_zone_secondary_y = DropZone('secondary_y')
        self.drop_zone_secondary_y.setMaximumWidth(80)
        self.drop_zone_secondary_y.dataset_dropped.connect(self.on_dataset_dropped)
        
        # Zone de drop inférieure (axe X)
        self.drop_zone_x = DropZone('x_axis')
        self.drop_zone_x.setMaximumHeight(60)
        self.drop_zone_x.dataset_dropped.connect(self.on_dataset_dropped)

        # Adding Widgets and layouts
        plot_container_layout.addWidget(self.plot_title)
        plot_layout.addWidget(toolbar)
        plot_layout.addWidget(self.drop_zone_secondary_x)
        middle_layout.addWidget(self.drop_zone_overlay)
        middle_layout.addWidget(self.plot_canvas)
        middle_layout.addWidget(self.drop_zone_secondary_y)
        plot_layout.addLayout(middle_layout)
        plot_layout.addWidget(self.drop_zone_x)
        plot_container_layout.addWidget(plot_with_drops)
        splitter.addWidget(plot_container)
        
        # Répartition de l'espace
        splitter.setSizes([350, 1050])
        
        # Message de bienvenue
        self.show_welcome_message()
        
    def show_welcome_message(self):
        """Affiche un message de bienvenue"""
        self.plot_canvas.axes.clear()
        self.plot_canvas.axes.text(0.5, 0.5, 
                                  "Explorateur HDF5 Multi-Axes\n\n" +
                                  "1. Ouvrez un fichier HDF5\n" +
                                  "2. Cliquez sur un dataset pour l'afficher\n" +
                                  "3. Glissez d'autres datasets vers les zones colorées:\n" +
                                  "   • Bas: Axe X personnalisé\n" +
                                  "   • Côtés: Superposition\n" +
                                  "   • Haut: Axe Y secondaire",
                                  transform=self.plot_canvas.axes.transAxes,
                                  ha='center', va='center', fontsize=11,
                                  bbox=dict(boxstyle="round,pad=0.5", facecolor="lightblue", alpha=0.7))
        self.plot_canvas.axes.set_xlim(0, 1)
        self.plot_canvas.axes.set_ylim(0, 1)
        self.plot_canvas.axes.axis('off')
        self.plot_canvas.draw()
        
    def open_file(self, file_path=None):
        """Ouvre une boîte de dialogue pour sélectionner un fichier HDF5"""
        if not file_path:
            file_path, _ = QFileDialog.getOpenFileName(
                self, "Ouvrir fichier HDF5", "", 
                "Fichiers NeXus (*.nxs *.nexus);;Fichiers HDF5 (*.h5 *.hdf5 *.he5);;Tous les fichiers (*)")
        
        if file_path:
            self.tree_widget.load_hdf5_file(file_path)
            self.plot_title.setText(f"Données multi-axes: {file_path.split('/')[-1]}")
            
    def on_dataset_selected(self, dataset, name):
        """Appelé quand un dataset est sélectionné dans l'arbre (clic simple)"""
        try:
            # The tree may emit either a Dataset or a Group (e.g. an NXdata group).
            # If it's a Group that represents NXdata (or a '/.../data' group),
            # resolve the linked data and axes and display them automatically.
            if isinstance(dataset, h5py.Group):
                grp = dataset
                nxclass = grp.attrs.get('NX_class', '')
                is_nxdata = False
                try:
                    if isinstance(nxclass, bytes):
                        is_nxdata = nxclass.decode().upper().startswith('NXDATA')
                    else:
                        is_nxdata = str(nxclass).upper().startswith('NXDATA')
                except Exception:
                    is_nxdata = False

                # also accept groups named 'data' as NXdata containers
                if grp.name.endswith('/data'):
                    is_nxdata = True

                if is_nxdata:
                    # try to find the main dataset inside the NXdata group
                    main_ds = None
                    # prefer attribute 'signal' if present
                    raw_sig = grp.attrs.get('signal') if 'signal' in grp.attrs else None
                    sig = None
                    if raw_sig is not None:
                        # Normalize various possible storage forms into a string
                        try:
                            if isinstance(raw_sig, (list, tuple)):
                                raw_sig = raw_sig[0]
                            if isinstance(raw_sig, np.ndarray):
                                # try to extract first element
                                try:
                                    raw_sig = raw_sig.tolist()
                                    if isinstance(raw_sig, (list, tuple)):
                                        raw_sig = raw_sig[0]
                                except Exception:
                                    # fallback to str conversion
                                    raw_sig = str(raw_sig)
                            if isinstance(raw_sig, bytes) or isinstance(raw_sig, np.bytes_):
                                try:
                                    sig = raw_sig.decode()
                                except Exception:
                                    sig = str(raw_sig)
                            else:
                                sig = str(raw_sig)
                        except Exception:
                            sig = None

                    if sig is not None and sig != '':
                        try:
                            # resolve absolute or relative
                            if isinstance(sig, str) and sig.startswith('/'):
                                main_ds = grp.file[sig]
                            elif sig in grp:
                                main_ds = grp[sig]
                        except Exception:
                            main_ds = None

                    # fallback: choose the first Dataset child (prefer ndim>1)
                    if main_ds is None:
                        for k, v in grp.items():
                            if isinstance(v, h5py.Dataset):
                                if getattr(v, 'ndim', 0) > 1:
                                    main_ds = v
                                    break
                                if main_ds is None:
                                    main_ds = v

                    # deep search if still not found
                    def _find_dataset(g):
                        for kk, vv in g.items():
                            if isinstance(vv, h5py.Dataset):
                                return vv
                            elif isinstance(vv, h5py.Group):
                                res = _find_dataset(vv)
                                if res:
                                    return res
                        return None

                    if main_ds is None:
                        main_ds = _find_dataset(grp)

                    if main_ds is None:
                        # not an NXdata we can handle; fall back to showing group content
                        self._display_as_content(grp, name)
                        return

                    # Prefer explicit 'axes' attribute on NXdata if present. Otherwise
                    # fall back to heuristic search within the same scan group.
                    parent_path = '/'.join(grp.name.split('/')[:-1])
                    try:
                        parent_group = grp.file[parent_path] if parent_path else grp.file['/']
                    except Exception:
                        parent_group = grp

                    def _resolve_axis_by_path(spec):
                        """Resolve an axis specifier which may be a path or a name.
                        Returns an h5py.Dataset or None.
                        """
                        if spec is None:
                            return None
                        try:
                            if isinstance(spec, bytes) or isinstance(spec, np.bytes_):
                                spec = spec.decode()
                        except Exception:
                            pass
                        try:
                            spec_str = str(spec)
                        except Exception:
                            return None

                        # absolute path
                        if spec_str.startswith('/'):
                            try:
                                obj = grp.file[spec_str]
                                return obj if isinstance(obj, h5py.Dataset) else None
                            except Exception:
                                return None

                        # relative to NXdata group
                        if spec_str in grp:
                            obj = grp[spec_str]
                            return obj if isinstance(obj, h5py.Dataset) else None

                        # relative to parent scan group
                        try:
                            if spec_str in parent_group:
                                obj = parent_group[spec_str]
                                return obj if isinstance(obj, h5py.Dataset) else None
                        except Exception:
                            pass

                        # not found
                        return None

                    axes_attr = grp.attrs.get('axes') if 'axes' in grp.attrs else None
                    axes_found = []
                    main_array = None
                    display_name = ''

                    try:
                        main_array = main_ds[:]
                        display_name = main_ds.name.split('/')[-1]

                        if axes_attr is not None:
                            # Normalize axes_attr into a list
                            resolved_axes = []
                            try:
                                if isinstance(axes_attr, (list, tuple, np.ndarray)):
                                    iterable = list(axes_attr)
                                else:
                                    iterable = [axes_attr]
                                for a in iterable:
                                    # some axes may be stored as references or bytes
                                    resolved = _resolve_axis_by_path(a)
                                    resolved_axes.append(resolved)
                            except Exception:
                                resolved_axes = []

                            # validate lengths against main_ds.shape
                            main_shape = getattr(main_ds, 'shape', ()) or ()
                            for i, ax in enumerate(resolved_axes):
                                if ax is None:
                                    axes_found.append(None)
                                    continue
                                try:
                                    ax_len = len(ax)
                                except Exception:
                                    ax_len = None
                                expected = main_shape[i] if i < len(main_shape) else None
                                if expected is None or ax_len == expected:
                                    axes_found.append(ax)
                                else:
                                    # length mismatch -> do not set this axis
                                    axes_found.append(None)

                        else:
                            # Fallback: collect 1D datasets within the parent scan group
                            candidates = []
                            def _collect_1d(g):
                                for kk, vv in g.items():
                                    if isinstance(vv, h5py.Dataset) and getattr(vv, 'ndim', 0) == 1:
                                        candidates.append((g.name + '/' + kk, vv))
                                    elif isinstance(vv, h5py.Group):
                                        _collect_1d(vv)
                            try:
                                _collect_1d(parent_group)
                            except Exception:
                                pass

                            # match axes by dimension lengths (prefer closest match)
                            for dim in getattr(main_ds, 'shape', []):
                                match = None
                                for i, (p, ds) in enumerate(candidates):
                                    try:
                                        if len(ds) == dim:
                                            match = (p, ds)
                                            candidates.pop(i)
                                            break
                                    except Exception:
                                        continue
                                axes_found.append(match[1] if match else None)

                        # assign axes to canvas: for 1D main datasets map axis -> x_axis
                        # for 2D main datasets map last axis -> x_axis (columns), first axis -> secondary_x (rows)
                        scatter_flag = False
                        if axes_found:
                            main_shape = getattr(main_ds, 'shape', None) or ()
                            # map last axis to x_axis if present 
                            last = axes_found[-1] if len(axes_found) >= 1 else None
                            if len(main_shape) == 1 and len(axes_found) == 1 and last is not None:
                                try:
                                    axis_len = len(last)
                                except Exception:
                                    axis_len = None
                                expected = main_shape[-1] if len(main_shape) >= 1 else None
                                if expected is None or axis_len == expected:
                                    self.plot_canvas.add_x_axis_dataset(last[:], last.name.split('/')[-1])
                            if len(main_shape) == 1 and len(axes_found) > 1 and last is not None:
                                # Multiple candidate axes found for a 1D main dataset.
                                # Let the user choose which two axes to use for a scatter plot.
                                try:
                                    # Build a list of non-None candidate axes and names
                                    candidates = [ax for ax in axes_found if ax is not None]
                                    names = [ax.name.split('/')[-1] for ax in candidates]

                                    if len(names) < 2:
                                        QMessageBox.information(self, "Axes insuffisants", "Pas assez d'axes candidats pour créer un scatter.")
                                    else:
                                        dlg = AxisSelectionDialog(names, parent=self)
                                        if dlg.exec_() == QDialog.Accepted:
                                            x_name, y_name = dlg.get_selection()
                                            # find corresponding datasets (match by last path component)
                                            x_ds = next((ax for ax in candidates if ax.name.split('/')[-1] == x_name), None)
                                            y_ds = next((ax for ax in candidates if ax.name.split('/')[-1] == y_name), None)

                                            if x_ds is None or y_ds is None:
                                                QMessageBox.warning(self, "Axe introuvable", "Les axes sélectionnés n'ont pas pu être résolus.")
                                            else:
                                                try:
                                                    x_vals = x_ds[:] if isinstance(x_ds, h5py.Dataset) else np.asarray(x_ds)
                                                except Exception:
                                                    x_vals = np.asarray(x_ds)
                                                try:
                                                    y_vals = y_ds[:] if isinstance(y_ds, h5py.Dataset) else np.asarray(y_ds)
                                                except Exception:
                                                    y_vals = np.asarray(y_ds)

                                                self.plot_canvas.clear_plot()
                                                # color the points by the main dataset values if shapes match, otherwise plain
                                                cvals = None
                                                try:
                                                    cvals = main_array
                                                    # flatten if necessary
                                                    if getattr(cvals, 'ndim', 1) > 1:
                                                        cvals = cvals.flatten()
                                                except Exception:
                                                    cvals = None

                                                try:
                                                    if cvals is not None and len(cvals) == len(x_vals) and len(cvals) == len(y_vals):
                                                        sc = self.plot_canvas.axes.scatter(x_vals, y_vals, c=cvals, s=10, cmap='viridis')
                                                        # create colorbar for scatter
                                                        try:
                                                            if getattr(self.plot_canvas, '_img_colorbar', None) is not None:
                                                                try:
                                                                    self.plot_canvas._img_colorbar.remove()
                                                                except Exception:
                                                                    pass
                                                            cax = self.plot_canvas.fig.add_axes([0.92, 0.11, 0.02, 0.77])
                                                            self.plot_canvas._img_colorbar = self.plot_canvas.fig.colorbar(sc, cax=cax)
                                                        except Exception:
                                                            pass
                                                    else:
                                                        self.plot_canvas.axes.scatter(x_vals, y_vals, s=10)

                                                    ux = x_ds.name.split('/')[-1]
                                                    uy = y_ds.name.split('/')[-1]
                                                    uxx = x_ds.attrs.get('units', '') if hasattr(x_ds, 'attrs') else ''
                                                    uyy = y_ds.attrs.get('units', '') if hasattr(y_ds, 'attrs') else ''
                                                    self.plot_canvas.axes.set_xlabel(f"{ux} ({uxx})")
                                                    self.plot_canvas.axes.set_ylabel(f"{uy} ({uyy})")
                                                    self.plot_canvas.axes.set_title(display_name)
                                                    self.plot_canvas.draw()
                                                    scatter_flag = True
                                                except Exception as e:
                                                    QMessageBox.warning(self, "Erreur scatter", f"Erreur lors de la création du scatter: {e}")
                                except Exception as e:
                                    QMessageBox.warning(self, "Erreur", f"Erreur lors de la sélection des axes: {e}")
                            # if 2D and first axis present, map to secondary_x (rows)
                            elif len(main_shape) == 2 and len(axes_found) > 1 and axes_found[0] is not None:
                                first = axes_found[0]
                                try:
                                    axis_len0 = len(first)
                                except Exception:
                                    axis_len0 = None
                                expected0 = main_shape[0]
                                if axis_len0 == expected0:
                                    self.plot_canvas.add_secondary_x_dataset(first[:], first.name.split('/')[-1])
                                self.plot_canvas.add_x_axis_dataset(last[:], last.name.split('/')[-1])
                                
                        if not scatter_flag:
                            self._display_as_plot(main_ds, main_array, display_name)
                        return
                    except Exception as e:
                        QMessageBox.warning(self, "Erreur", f"Impossible d'afficher NXdata: {e}")

            # Not an NXdata group: try to treat dataset-like objects
            data = dataset
            
            # Vérifie si les données peuvent être affichées en graphique
            can_plot = self._can_plot_data(data)
            
            if can_plot:
                # Affichage graphique normal
                res = re.match("(?P<name>[\w\s]+)\s*\((?P<shape>[\d\s,]+)\)\s*\((?P<type>[\w\s]+)\)\s*$", name)
                unit = data.attrs.get('units') if hasattr(data, 'attrs') else None
                if res:
                    name = f"{res.group('name')} ({unit}) [shape=({res.group('shape')}), type={res.group('type')}]"
                self._display_as_plot(dataset, data[:], name)
            else:
                # Affichage du contenu dans une fenêtre séparée
                self._display_as_content(data, name)
                
        except Exception as e:
            QMessageBox.warning(self, "Erreur", f"Impossible de charger le dataset:\n{str(e)}")
            
    def _can_plot_data(self, data):
        """Détermine si les données peuvent être affichées en graphique"""
        try:
            data = data[:]
            # Données numériques avec dimensions appropriées
            if hasattr(data, 'dtype'):
                # Types numériques
                if np.issubdtype(data.dtype, np.number):
                    # Dimensions 1D ou 2D avec taille raisonnable
                    if data.ndim <= 2 and data.size > 0:
                        return True
                        
                # Types string/bytes mais petits (pourraient contenir des nombres)
                if data.dtype.kind in ['S', 'U'] and data.size < 10:
                    # Essaie de convertir en numérique
                    try:
                        if data.ndim == 0:  # scalaire
                            float(data.item())
                            return True
                        else:
                            test_data = data.flatten()[:5]  # teste quelques valeurs
                            [float(x) for x in test_data if x.strip()]
                            return True
                    except (ValueError, AttributeError):
                        pass
                        
            return False
            
        except Exception:
            return False
            
    def _display_as_plot(self, dataset, data, name):
        """Affiche les données comme un graphique"""
        # Met à jour les informations
        info_text = f"Dataset principal: {name}\n"
        info_text += f"Shape: {dataset.shape}\n"
        info_text += f"Dtype: {dataset.dtype}\n"
        info_text += f"Taille: {dataset.size} éléments\n"
        
        if len(dataset.attrs) > 0:
            info_text += "\nAttributs:\n"
            for attr_name, attr_value in dataset.attrs.items():
                info_text += f"  • {attr_name}: {attr_value}\n"
                
        # Ajoute les informations des autres datasets
        if self.plot_canvas.datasets['x_axis'] is not None:
            info_text += f"\nAxe X: {self.plot_canvas.dataset_names['x_axis']}\n"
        if len(self.plot_canvas.datasets['overlay']) > 0:
            info_text += f"Superpositions: {len(self.plot_canvas.datasets['overlay'])}\n"
        if len(self.plot_canvas.datasets['secondary_y']) > 0:
            info_text += f"Axe Y2: {len(self.plot_canvas.datasets['secondary_y'])}\n"
        if self.plot_canvas.datasets['secondary_x'] is not None:
            # secondary_x is a 1D array (or None); show its name if available
            name = self.plot_canvas.dataset_names.get('secondary_x', '')
            info_text += f"Axe X2: {name}\n"
                
        self.info_label.setText(info_text)
        
        # Définit comme dataset principal
        self.plot_canvas.set_main_dataset(data, name)
        
    def _display_as_content(self, data, name):
        """Affiche le contenu dans une fenêtre séparée"""
        # Met à jour les informations dans la barre latérale
        info_text = f"Dataset (contenu texte): {name}\n"
        
        if hasattr(data, 'shape'):
            info_text += f"Shape: {data.shape}\n"
        if hasattr(data, 'dtype'):
            info_text += f"Dtype: {data.dtype}\n"
        if hasattr(data, 'size'):
            info_text += f"Taille: {data.size} éléments\n"
        if data.attrs.get('units') is not None:
            info_text += f"Unité: {data.attrs.get('units')}"
            
        info_text += "\n→ Contenu affiché dans une fenêtre séparée"
        self.info_label.setText(info_text)
        
        # Ouvre la fenêtre de visualisation du contenu
        viewer = DataViewer(data, name, self)
        viewer.show()
        
        # Affiche un message informatif dans la zone de plot
        self.plot_canvas.axes.clear()
        self.plot_canvas.axes.text(0.5, 0.5, 
                                  f"Dataset non-graphique:\n'{name}'\n\n" +
                                  "Le contenu est affiché dans\nune fenêtre séparée",
                                  transform=self.plot_canvas.axes.transAxes,
                                  ha='center', va='center', fontsize=12,
                                  bbox=dict(boxstyle="round,pad=0.3", facecolor="lightyellow", alpha=0.8))
        self.plot_canvas.axes.set_xlim(0, 1)
        self.plot_canvas.axes.set_ylim(0, 1)
        self.plot_canvas.axes.axis('off')
        self.plot_canvas.draw()
            
    def on_dataset_dropped(self, dataset, name, zone_type):
        """Appelé quand un dataset est déposé dans une zone"""
        try:
            data = dataset[:]
            
            if zone_type == 'x_axis':
                self.plot_canvas.add_x_axis_dataset(data, name)
                QMessageBox.information(self, "Dataset ajouté", 
                                      f"'{name}' défini comme axe X")
            elif zone_type == 'overlay':
                self.plot_canvas.add_overlay_dataset(data, name)
                QMessageBox.information(self, "Dataset ajouté", 
                                      f"'{name}' ajouté en superposition")
            elif zone_type == 'secondary_y':
                self.plot_canvas.add_secondary_y_dataset(data, name)
                QMessageBox.information(self, "Dataset ajouté", 
                                      f"'{name}' ajouté sur l'axe Y secondaire")
            elif zone_type == 'secondary_x':
                self.plot_canvas.add_secondary_x_dataset(data, name)
                QMessageBox.information(self, "Dataset ajouté", 
                                      f"'{name}' ajouté sur l'axe X secondaire")
                                      
        except Exception as e:
            QMessageBox.warning(self, "Erreur", f"Impossible d'ajouter le dataset:\n{str(e)}\n\n {traceback.format_exc()}")
            
    def clear_plot(self):
        """Efface le graphique et tous les datasets"""
        self.plot_canvas.clear_plot()
        self.info_label.setText("Graphique effacé - Sélectionnez un dataset pour recommencer")
        self.show_welcome_message()
        
    def export_to_csv(self):
        """Lance l'export des données vers un fichier CSV"""
        if self.plot_canvas.datasets['main'] is None:
            QMessageBox.warning(self, "Aucune donnée", 
                              "Aucune donnée à exporter. Sélectionnez d'abord un dataset.")
            return
            
        # Boîte de dialogue pour choisir le fichier de sauvegarde
        file_path, _ = QFileDialog.getSaveFileName(
            self, "Exporter les données en CSV", 
            "donnees_hdf5_export.csv",
            "Fichiers CSV (*.csv);;Tous les fichiers (*)")
        
        if file_path:
            success, message = self.plot_canvas.export_to_csv(file_path)
            
            if success:
                QMessageBox.information(self, "Export réussi", 
                                      f"Les données ont été exportées avec succès!\n\n{message}")
            else:
                QMessageBox.critical(self, "Erreur d'export", message)
        
    def closeEvent(self, event):
        """Gère la fermeture de l'application"""
        if hasattr(self.tree_widget, 'hdf5_file') and self.tree_widget.hdf5_file:
            self.tree_widget.hdf5_file.close()
        event.accept()


def main(filename=None):
    """Fonction principale"""
    app = QApplication(sys.argv)
    
    # Style de l'application
    app.setStyleSheet("""
        QMainWindow {
            background-color: #f5f5f5;
        }
        QPushButton {
            background-color: #4CAF50;
            color: white;
            border: none;
            padding: 8px;
            border-radius: 4px;
            font-weight: bold;
        }
        QPushButton:hover {
            background-color: #45a049;
        }
        QTreeWidget {
            background-color: white;
            border: 1px solid #ddd;
            border-radius: 4px;
        }
        QGroupBox {
            font-weight: bold;
            margin-top: 10px;
        }
        QGroupBox::title {
            subcontrol-origin: margin;
            left: 10px;
            padding: 0 5px;
        }
    """)
    
    explorer = HDF5Explorer()
    explorer.open_file(file_path=filename)
    explorer.show()
    
    sys.exit(app.exec_())


if __name__ == "__main__":
    import sys
    import argparse
    commandParser = argparse.ArgumentParser(description="inline file open")
    commandParser.add_argument("-f", "--file", help="File to open")
    args = commandParser.parse_args()

    main(args.file)