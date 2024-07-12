# FETransNet
A work about multi-objective segmentation (Medical image)


![Net](https://github.com/user-attachments/assets/e5890a70-3844-4f68-9fa9-91895cd7c64a)
Fig.1 Overview of FETransNet


Quantitative results.

\begin{table}\centering
\begin{tabular}{c|c|cc|cccccccc}
\hline
\makebox[0.01\textwidth][c]{Methods}  & 
\makebox[0.01\textwidth][c]{Year} &
\makebox[0.015\textwidth][c]{DSC$\uparrow$} &
\makebox[0.015\textwidth][c]{HD$\downarrow$}
& \makebox[0.05\textwidth][c]{Aorta} &
\makebox[0.09\textwidth][c]{Gallbladdr} &
\makebox[0.09\textwidth][c]{Kidney(L)} &
\makebox[0.08\textwidth][c]{Kidney(R)}
& \makebox[0.07\textwidth][c]{Liver} &
\makebox[0.07\textwidth][c]{Pancreas} &
\makebox[0.07\textwidth][c]{Spleen} &
\makebox[0.07\textwidth][c]{Stomach}
\\
\hline
U-Net   & 2015 & 76.85 & 39.70 & 
89.07 & 69.72 & 77.77 & 68.60 & 
93.43 & 53.98 & 86.67 & 75.58 \\
AttUNet  & 2018 & 77.77 & 36.02 & 
\textbf{89.55} & 68.88 & 77.98 & 71.11 & 
93.57 & 58.04 & 87.30 & 75.75 \\
R50 ViT  & 2020 & 71.29 & 32.87 & 
73.73 & 55.13 & 75.80 & 72.20 & 
91.51 & 45.99 & 81.99 & 73.95 \\
TransUnet  & 2021 & 77.48 & 31.69 & 
87.23 & 63.13 & 81.87 & 77.02 & 
94.08 & 55.86 & 85.08 & 75.62 \\
UCTransNet   & 2021 & 78.23 & 26.74 & 
88.86 & 66.97 & 80.18 & 73.17 & 
93.16 & 56.22 & 87.84 & 79.43 \\
SwinUNet   & 2021 & 79.12 & 21.55 & 
85.47 & 66.53 & 83.28 & 79.61 & 
94.29 & 56.58 & 90.66 & 76.60 \\
MISSFormer  & 2021 & 81.96 & 18.20 & 
86.99 & 68.65 & 85.21 & 82.00 & 
94.41 & 65.67 & 91.92 & 80.81 \\
HiFormer  & 2022 & 80.69 & 19.14 & 
87.03 & 68.61 & 84.23 & 78.37 & 
94.07 & 60.77 & 90.44 & 82.03 \\
ScaleFormer   & 2022 & 82.86 & 16.81 & 
88.73 & 74.97 & 86.36 & 83.31 & 
95.12 & 64.85 & 89.40 & 80.14 \\
\hline
FETransNet & Ours & \textbf{85.50} & \textbf{12.18} &
89.14 & \textbf{75.77} & \textbf{89.10} & \textbf{86.39} & \textbf{95.62} & \textbf{71.42} & \textbf{93.10} & \textbf{83.47} \\
\hline
\end{tabular}
\label{tab:synapse}
\end{table}

\begin{table}[H]\centering
\begin{tabular}{c|c|cc}
\hline
\makebox[0.01\textwidth][c]{Methods}  & \makebox[0.01\textwidth][c]{Year} & \makebox[0.01\textwidth][c]{DSC$\uparrow$} & \makebox[0.01\textwidth][c]{IOU$\uparrow$}
\\
\hline
U-Net++  & 2018 & 75.28 & 60.89 \\
AttUNet  & 2018 & 76.20 & 62.64 \\
MRUNet   & 2020 & 77.54 & 63.80 \\
MedT   & 2021 & 79.24 & 65.73 \\
TransUNet   & 2021 & 79.20 & 65.68 \\
SwinUNet  & 2021 & 78.49 & 64.72 \\
UCTransNet  & 2021 & 79.87 & 66.68 \\
ScaleFormer   & 2022 & 80.06 & 66.87 \\
\hline
FETransNet & Ours & \textbf{80.44} & \textbf{67.41} \\
\hline
\end{tabular}
\label{tap:monuseg}
\end{table}


Add some visual results.
![Synapse](https://github.com/user-attachments/assets/7b7e43f3-9d96-41d8-bc7c-a70e74dc3d6e)
![MoNuSeg](https://github.com/user-attachments/assets/d8fe495b-d547-40c7-bc69-c0e249d96862)
