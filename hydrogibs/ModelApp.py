import customtkinter as ctk
from typing import Tuple, Literal

from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg,\
    NavigationToolbar2Tk
from matplotlib.backend_bases import key_press_handler
from matplotlib.pyplot import close
import customtkinter as ctk

from hydrogibs.ModelTemplate import Catchment, Rain


class Entry:

    def __init__(self,
                 object: Literal["catchment", "rain"],
                 key: str,
                 unit: str,
                 alias: str = None) -> None:
        self.object = object
        self.key = key
        self.unit = unit
        self.alias = alias


class ModelApp:

    def __init__(self,
                 catchment: Catchment,
                 rain: Rain,
                 entries: Tuple[Entry],
                 title: str = None,
                 appearance: str = "dark",
                 color_theme: str = "dark-blue",
                 style: str = "seaborn",
                 close_and_clear: bool = True,
                 *args, **kwargs):

        self.catchment = catchment
        self.rain = rain
        self.event = rain @ catchment

        ctk.set_appearance_mode(appearance)
        ctk.set_default_color_theme(color_theme)

        self.root = ctk.CTk()
        self.root.title(title)
        self.root.bind('<Return>', self.entries_update)

        self.dframe = ctk.CTkFrame(master=self.root)
        self.dframe.grid(row=0, column=1, sticky="NSEW")

        self.init_diagram(style=style, show=False, *args, **kwargs)

        self.pframe = ctk.CTkFrame(master=self.root)
        self.pframe.grid(column=0, row=0, sticky="NSEW")

        self.entry_row = 1
        self.entries = dict()
        for entry in entries:
            self.define_entry(entry)

        ctk.CTkButton(master=self.pframe,
                      text="Reset zoom",
                      command=lambda: self.diagram.zoom(self.canvas)
                      ).grid(pady=10)

        self.root.mainloop()
        if close_and_clear:
            close()

    def init_diagram(self, *args, **kwargs):

        diagram = self.event.diagram(*args, **kwargs)

        self.canvas = FigureCanvasTkAgg(diagram.figure, master=self.dframe)
        toolbar = NavigationToolbar2Tk(canvas=self.canvas, window=self.dframe)
        toolbar.update()
        self.canvas._tkcanvas.pack()
        self.canvas.draw()
        self.canvas.get_tk_widget().pack()
        self.canvas.mpl_connect('key_press_event',
                                lambda arg: key_press_handler(
                                    arg, self.canvas, toolbar
                                ))
        self.diagram = diagram
        self.root.update()

    def define_entry(self, entry: Entry):

        erow = self.entry_row
        object = entry.object
        key = entry.key
        unit = entry.unit
        alias = entry.alias

        entryframe = ctk.CTkFrame(master=self.pframe)
        entryframe.grid(sticky="NSEW")
        unit_str = f"[{unit}]"
        name = key if alias is None else alias
        label = ctk.CTkLabel(master=entryframe,
                             text=f"{name:>5} {unit_str:>6} ",
                             font=("monospace", 14))
        label.grid(row=erow, column=0, sticky="EW", ipady=5)

        input = ctk.CTkEntry(master=entryframe)

        if object == "catchment":
            value = getattr(self.catchment, key)
        elif object == "rain":
            value = getattr(self.rain, key)
        else:
            raise KeyError(f"{key} not an object")
        input.insert(0, value)

        input.grid(row=erow, column=1, sticky="EW")

        slider = ctk.CTkSlider(
            master=entryframe,
            from_=0, to=2*value if value else 1,
            number_of_steps=999,
            command=lambda _: self.slider_update(object, key)
        )
        slider.grid(row=erow, column=2, sticky="EW")

        self.entries[key] = dict(
            object=object,
            label=label,
            input=input,
            slider=slider
        )

        self.entry_row = erow + 1

    def slider_update(self, object: str, key: str):

        value = self.entries[key]["slider"].get()
        self.entries[key]["input"].delete(0, ctk.END)
        self.entries[key]["input"].insert(0, f"{value:.2f}")
        self.update_attribute(object, key, value)
        self.update()

    def entries_update(self, _):

        for key in self.entries:

            entry = self.entries[key]
            value = float(entry['input'].get())
            self.update_attribute(entry["object"], key, value)

            v = value if value else 1
            slider = entry["slider"]
            slider.configure(to=2*v)
            slider.set(v)

        self.update()

    def update_attribute(self, object: str, key, value):

        if object == "catchment":
            setattr(self.catchment, key, value)
        elif object == "rain":
            setattr(self.rain, key, value)
        else:
            raise KeyError(f"{key} not an attribute ('catchment' or 'rain')")

    def update(self):

        event = self.rain @ self.catchment
        self.diagram.update(event)
        self.canvas.draw()
