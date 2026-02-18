# import tkinter as tk
#
# def main():
#     # 其他初始化代码
#     print("Found 7 age indices and 7 sph_r indices")
#     print("Target age index: 17, Target sph index: None")
#
#     # 模拟多轮运行
#     for simulation in range(2):
#         for step in range(3):
#             predicted_value = -1.9015 + (step * 0.3025)
#             new_age = 11.50 + (step * 0.50)
#             print(f"Simulation {simulation}, Step {step}:")
#             print(f"Predicted value: {predicted_value:.4f}")
#             print(f"New age: {new_age:.2f}")
#
#     # Tkinter调用（仅作为示例）
#     root = tk.Tk()
#     root.title("Test Window")
#     label = tk.Label(root, text="Hello, Tkinter!")
#     label.pack(pady=20)
#     root.mainloop()
#
# if __name__ == "__main__":
#     try:
#         main()
#     except Exception as e:
#         print(f"预测过程中出现错误: {e}")



import matplotlib.font_manager

# 获取所有可用字体
font_list = sorted({f.name for f in matplotlib.font_manager.fontManager.ttflist})

for font in font_list:
    print(font)

"""
Agency FB
Algerian
AlienCaret
Arial
Arial Rounded MT Bold
Bahnschrift
Baskerville Old Face
Bauhaus 93
Bell MT
Berlin Sans FB
Berlin Sans FB Demi
Bernard MT Condensed
Blackadder ITC
Bodoni MT
Book Antiqua
Bookman Old Style
Bookshelf Symbol 7
Bradley Hand ITC
Britannic Bold
Broadway
Brush Script MT
Calibri
Californian FB
Calisto MT
Cambria
Candara
Cascadia Code
Cascadia Mono
Castellar
Centaur
Century
Century Gothic
Century Schoolbook
Chiller
Colonna MT
Comic Sans MS
Consolas
Constantia
Cooper Black
Copperplate Gothic Bold
Copperplate Gothic Light
Corbel
Courier New
Curlz MT
DejaVu Math TeX Gyre
DejaVu Sans
DejaVu Sans Display
DejaVu Sans Mono
DejaVu Serif
DejaVu Serif Display
DengXian
Dubai
Ebrima
Edwardian Script ITC
Elephant
Engravers MT
Eras Bold ITC
Eras Demi ITC
Eras Light ITC
Eras Medium ITC
FZShuTi
FZYaoTi
FangSong
Felix Titling
Footlight MT Light
Forte
Franklin Gothic Book
Franklin Gothic Demi
Franklin Gothic Demi Cond
Franklin Gothic Heavy
Franklin Gothic Medium
Franklin Gothic Medium Cond
Freestyle Script
French Script MT
Gabriola
Gadugi
Garamond
Georgia
Gigi
Gill Sans MT
Gill Sans MT Condensed
Gill Sans MT Ext Condensed Bold
Gill Sans Ultra Bold
Gill Sans Ultra Bold Condensed
Gloucester MT Extra Condensed
Goudy Old Style
Goudy Stout
Haettenschweiler
Harlow Solid Italic
Harrington
High Tower Text
HoloLens MDL2 Assets
Impact
Imprint MT Shadow
Informal Roman
Ink Free
Javanese Text
Jokerman
Juice ITC
KaiTi
Kingsoft Symbol
Kristen ITC
Kunstler Script
Leelawadee
Leelawadee UI
LiSu
Lucida Bright
Lucida Calligraphy
Lucida Console
Lucida Fax
Lucida Handwriting
Lucida Sans
Lucida Sans Typewriter
Lucida Sans Unicode
MS Gothic
MS Outlook
MS Reference Sans Serif
MS Reference Specialty
MT Extra
MV Boli
Magneto
Maiandra GD
Malgun Gothic
Matura MT Script Capitals
Microsoft Himalaya
Microsoft JhengHei
Microsoft New Tai Lue
Microsoft PhagsPa
Microsoft Sans Serif
Microsoft Tai Le
Microsoft Uighur
Microsoft YaHei
Microsoft Yi Baiti
MingLiU-ExtB
Mistral
Modern No. 20
Mongolian Baiti
Monotype Corsiva
Myanmar Text
Niagara Engraved
Niagara Solid
Nirmala UI
OCR A Extended
Old English Text MT
Onyx
Open Sans
Palace Script MT
Palatino Linotype
Papyrus
Parchment
Perpetua
Perpetua Titling MT
Pill Alt 300mg
Pill Alt 600mg
Playbill
Poor Richard
Pristina
Rage Italic
Ravie
Rockwell
Rockwell Condensed
Rockwell Extra Bold
STCaiyun
STFangsong
STHupo
STIXGeneral
STIXNonUnicode
STIXSizeFiveSym
STIXSizeFourSym
STIXSizeOneSym
STIXSizeThreeSym
STIXSizeTwoSym
STKaiti
STLiti
STSong
STXihei
STXingkai
STXinwei
STZhongsong
Sans Serif Collection
Script MT Bold
Segoe Fluent Icons
Segoe MDL2 Assets
Segoe Print
Segoe Script
Segoe UI
Segoe UI Emoji
Segoe UI Historic
Segoe UI Symbol
Segoe UI Variable
Showcard Gothic
SimHei
SimSun
SimSun-ExtB
Sitka
Snap ITC
Stencil
Sylfaen
Symbol
Tahoma
Tempus Sans ITC
Times New Roman
Trebuchet MS
Tw Cen MT
Tw Cen MT Condensed
Tw Cen MT Condensed Extra Bold
Verdana
Viner Hand ITC
Vivaldi
Vladimir Script
Webdings
Wide Latin
Wingdings
Wingdings 2
Wingdings 3
YouYuan
Yu Gothic
cmb10
cmex10
cmmi10
cmr10
cmss10
cmsy10
cmtt10
"""