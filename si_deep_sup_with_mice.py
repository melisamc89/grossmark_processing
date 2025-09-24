import pandas as pd
import numpy as np
import pickle as pkl
from src.config import *
from src.channel_information import *
neural_data_dir = files_directory
###############################################
#
#       PLOTTING SI VALUES VS FILTERS       #
#
#############################################

data_output_directory = '/home/melma31/Documents/time_project/SI_filters'
rats = [0]
sessions = [[0]]
probes_used = [[['Probe1','Probe2']]]

#rats = []
import os
import pickle as pkl
import pandas as pd
# Initialize list to collect rows
df_rows = []
# Lop through the rat data structure
for rat_index in rats:
    rat_name = rat_names[rat_index]
    for count_session, session_index in enumerate(sessions[rat_index]):
        for probe in probes_used[rat_index][count_session]:
            # Construct filename (assuming output_filename uses rat_name and session)
            output_filename = rat_name + '_' + rat_sessions[rat_name][session_index] + '_' + probe + '_si_100_2-60.pkl'
            file_path = os.path.join(data_output_directory, output_filename)
            # Load the .pkl dictionary
            with open(file_path, 'rb') as file:
                si_dict = pkl.load(file)
            # Access the data for this session and probe
            # Loop over behavioral labels and filter sizes
            for beh_label, filter_data in si_dict.items():
                for filter_size, data in filter_data.items():
                    row = {
                        'rat': rat_name,
                        'session': session_index,
                        'probe': probe,
                        'filter': int(filter_size),
                        'behavioral_label': beh_label,
                        'si': data['si'],
                        'si_deep': data['si_deep'],
                        'si_sup': data['si_sup'],
                        'time': int(filter_size) / 40,

                        # Optional: include behavior params too
                        # 'beh_params': data['beh_params']
                    }
                    df_rows.append(row)

#### ADD TERESA'S
#mouse_names = ['Calb','Thy']

mouse_names = ['Calb01_11_03_16_02_25_time_project_deep_sup.mat',
               'Calb01uLED_10_17_09_05_56_time_project_deep_sup.mat',
        'Calb01uLED_10_18_11_23_15_time_project_deep_sup.mat',
               'Calb01uLED_10_19_11_25_09_time_project_deep_sup.mat',
        'Calb01uLED_10_21_12_48_35_time_project_deep_sup.mat',
        'Thy01uLED_10_19_09_07_25_time_project_deep_sup.mat',
        'Thy01uLED_10_10_09_10_26_time_project_deep_sup.mat']

for mouse in mouse_names:
    output_filename = f"{mouse}_mouse_new_si_50_4-12_assimetry.pkl"
    #output_filename = f"{mouse}_mouse_si_100_2-60.pkl"
    file_path = os.path.join(data_output_directory, output_filename)
    # Load the .pkl dictionary
    with open(file_path, 'rb') as file:
        si_dict = pkl.load(file)
    # Access the data for this session and probe
    # Loop over behavioral labels and filter sizes
    for beh_label, filter_data in si_dict.items():
        for filter_size, data in filter_data.items():
            row = {
                'rat': mouse,
                'session': 1,
                'probe': 1,
                'filter': int(filter_size),
                'behavioral_label': beh_label,
                'si': data['si'],
                'si_deep': data['si_deep'],
                'si_sup': data['si_sup'],
                'time':int(filter_size)/20,
            }
            df_rows.append(row)

# Create DataFrame
df = pd.DataFrame(df_rows)
# Optional: Sort for readability
df.sort_values(by=['rat', 'session', 'probe', 'behavioral_label', 'filter'], inplace=True)
# Preview
print(df.head())

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
# Set plotting style
sns.set(style="whitegrid")

# Melt your DataFrame to long format to handle si, si_deep, si_sup as "Area"
df_long = df.melt(id_vars=['rat', 'session', 'probe', 'time', 'behavioral_label'],
                  value_vars=['si', 'si_deep', 'si_sup'],
                  var_name='Area', value_name='SI')

# Map custom colors and area names
area_palette = {
    'si': '#bbbcc0ff',
    'si_deep': '#cc9900',
    'si_sup': '#9900ff',
}


area_labels = {
    'si': 'Overall',
    'si_deep': 'Deep',
    'si_sup': 'Superficial'
}

# Create subplots
behavior_labels = df_long['behavioral_label'].unique()
fig, axes = plt.subplots(len(behavior_labels), 1, figsize=(8, 2 * len(behavior_labels)), sharex=True)

if len(behavior_labels) == 1:
    axes = [axes]

for i, beh in enumerate(behavior_labels):
    ax = axes[i]

    # Subset for this behavior
    sub_df = df_long[df_long['behavioral_label'] == beh]

    # Lineplot with shaded error band (std)
    sns.lineplot(data=sub_df, x='time', y='SI', hue='Area',
                 ax=ax, palette=area_palette, marker='o')

    # Formatting
    ax.set_title(f'Structure Index vs Filter Size — {beh}')
    ax.set_ylabel('Structure Index (SI)')
    ax.set_ylim(0, 1)
    ax.grid(False)

    if i == len(behavior_labels) - 1:
        ax.set_xlabel('Filter Size')
    else:
        ax.set_xlabel('')

    # Legend with friendly names
    handles, labels = ax.get_legend_handles_labels()
    new_labels = [area_labels.get(lbl, lbl) for lbl in labels]
    ax.legend(handles=handles, labels=new_labels, title='Area')

# Layout
#fig.suptitle('Structure Index vs Filter Size (Mean ± SD)', fontsize=16)
plt.tight_layout(rect=[0, 0, 1, 0.97])
plt.show()
fig.savefig(os.path.join(figures_directory, 'Si_rats_a_mice_100nei.png'))
fig.savefig(os.path.join(figures_directory, 'Si_rats_a_mice_100nei.svg'))

##### convert to time
import numpy as np

# Copy the original DataFrame to avoid modifying it globally
df_long_time = df_long.copy()

# Convert 'filter' to time in seconds: filter_size * (1 / 40)
#df_long_time['time'] = df_long_time['filter'] * (1 / 20)

# Filter to include only time <= 1.5s
df_long_time = df_long_time[df_long_time['time'] <= 1.5]

# Recreate the plot with time on x-axis
behavior_labels = df_long_time['behavioral_label'].unique()
fig, axes = plt.subplots(len(behavior_labels), 1, figsize=(8, 2 * len(behavior_labels)), sharex=True)

if len(behavior_labels) == 1:
    axes = [axes]

for i, beh in enumerate(behavior_labels):
    ax = axes[i]

    # Subset for this behavior
    sub_df = df_long_time[df_long_time['behavioral_label'] == beh]

    # Lineplot with shaded error band (std)
    sns.lineplot(data=sub_df, x='time', y='SI', hue='Area',
                 ax=ax, palette=area_palette, marker='o')

    # Formatting
    ax.set_title(f'Structure Index vs Time — {beh}')
    ax.set_ylabel('Structure Index (SI)')
    ax.set_ylim(0, 1)
    ax.grid(False)

    if i == len(behavior_labels) - 1:
        ax.set_xlabel('Time (s)')
    else:
        ax.set_xlabel('')

    # Legend with friendly names
    handles, labels = ax.get_legend_handles_labels()
    new_labels = [area_labels.get(lbl, lbl) for lbl in labels]
    ax.legend(handles=handles, labels=new_labels, title='Area')

# Layout
plt.tight_layout(rect=[0, 0, 1, 0.97])
plt.show()


#### plot only pos and time, and


import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from scipy.stats import ttest_ind
import os

# === INPUTS ===
output_dir = './figures'  # Change to your desired output directory
os.makedirs(output_dir, exist_ok=True)

# Area palette and label mapping
area_palette = {
    'si': '#bbbcc0ff',
    'si_deep': '#cc9900',
    'si_sup': '#9900ff'
}
area_labels = {
    'si': 'Overall',
    'si_deep': 'Deep',
    'si_sup': 'Superficial'
}
label_to_pos = {
    'Overall': 0,
    'Deep': 1,
    'Superficial': 2
}

# === STEP 1: Filter data at 0.4 s (filter size 16)
df_filtered = df[(df['time'] == 0.2) & (df['behavioral_label'].isin(['pos', 'time']))]

# === STEP 2: Melt to long format for plotting
df_long = pd.melt(
    df_filtered,
    id_vars=['rat', 'session', 'probe', 'behavioral_label'],
    value_vars=['si', 'si_deep', 'si_sup'],
    var_name='area',
    value_name='SI'
)
df_long['area_label'] = df_long['area'].map(area_labels)

# === STEP 3: Plot and add all pairwise t-tests
for beh in ['pos', 'time']:
    fig, ax = plt.subplots(figsize=(7, 5))
    df_beh = df_long[df_long['behavioral_label'] == beh]

    # Barplot
    sns.barplot(
        data=df_beh,
        x='area_label',
        y='SI',
        palette={area_labels[k]: v for k, v in area_palette.items()},
        capsize=0.1,
        ax=ax
    )

    # Stripplot of individual points
    sns.stripplot(
        data=df_beh,
        x='area_label',
        y='SI',
        color='black',
        alpha=0.4,
        size=5,
        jitter=True,
        dodge=False,
        ax=ax
    )

    # === Pairwise t-tests: all pairs
    pairs = [
        ('Overall', 'Deep'),
        ('Overall', 'Superficial'),
        ('Deep', 'Superficial')
    ]
    y_max = df_beh['SI'].max()
    increment = 0.05
    alpha_levels = [(0.001, '***'), (0.01, '**'), (0.05, '*')]

    for idx, (label1, label2) in enumerate(pairs):
        vals1 = df_beh.loc[df_beh['area_label'].eq(label1), 'SI'].dropna().values
        vals2 = df_beh.loc[df_beh['area_label'].eq(label2), 'SI'].dropna().values
        stat, pval = ttest_ind(vals1, vals2, equal_var=False)

        #stat, pval = ttest_ind(vals2, vals1, equal_var=False, alternative='greater')

        signif = next((symbol for alpha, symbol in alpha_levels if pval < alpha), 'n.s.')

        # Plot annotation if significant
        if signif != 'n.s.':
            x1, x2 = label_to_pos[label1], label_to_pos[label2]
            y = y_max + (idx + 1) * increment
            ax.plot([x1, x1, x2, x2], [y, y + 0.01, y + 0.01, y], lw=1.3, color='black')
            ax.text((x1 + x2) / 2, y + 0.015, signif, ha='center', va='bottom', fontsize=12)

    # Labels and formatting
    ax.set_title(f"Structure Index (SI) for '{beh}' at Filter Time = 0.2 s")
    ax.set_ylabel('Structure Index (SI)')
    ax.set_xlabel('Region')
    ax.set_ylim(0, y_max + 0.25)
    sns.despine()
    plt.tight_layout()

    # Save
    base_name = f"SI_bar_{beh}_filter_0.2s_allstats_2"
    png_path = os.path.join(output_dir, base_name + ".png")
    svg_path = os.path.join(output_dir, base_name + ".svg")
    fig.savefig(png_path, dpi=300, bbox_inches="tight")
    fig.savefig(svg_path, dpi=300, bbox_inches="tight")
    print(f"Saved plot for '{beh}' to:\n  - {png_path}\n  - {svg_path}")
    plt.show()
    plt.close()


###########################################################################################
##########################################################################################
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import ttest_ind, ttest_rel

# === INPUTS ===
output_dir = './figures'
os.makedirs(output_dir, exist_ok=True)

# Area palette and label mapping
area_palette = {
    'si': '#bbbcc0ff',
    'si_deep': '#cc9900',
    'si_sup': '#9900ff'
}
area_labels = {
    'si': 'Overall',
    'si_deep': 'Deep',
    'si_sup': 'Superficial'
}
label_to_pos = {
    'Overall': 0,
    'Deep': 1,
    'Superficial': 2
}

# === STEP 1: Filter data at 0.2 s
df_filtered = df[(df['time'] == 0.2) & (df['behavioral_label'].isin(['pos', 'time']))]

# === STEP 2: Melt to long format for plotting
df_long = pd.melt(
    df_filtered,
    id_vars=['rat', 'session', 'probe', 'behavioral_label'],
    value_vars=['si', 'si_deep', 'si_sup'],
    var_name='area',
    value_name='SI'
)
df_long['area_label'] = df_long['area'].map(area_labels)

# === STEP 3: Plot and add statistics
alpha_levels = [(0.001, '***'), (0.01, '**'), (0.05, '*')]

for beh in ['pos', 'time']:
    fig, ax = plt.subplots(figsize=(7, 5))
    df_beh = df_long[df_long['behavioral_label'] == beh]

    # Barplot
    sns.barplot(
        data=df_beh,
        x='area_label',
        y='SI',
        palette={area_labels[k]: v for k, v in area_palette.items()},
        capsize=0.1,
        ax=ax,
        #errorbar=("sd", 1.0)  # <-- use standard deviation
    )

    # Stripplot for all rats except Achilles
    sns.stripplot(
        data=df_beh[df_beh['rat'] != 'achilles'],
        x='area_label',
        y='SI',
        color='black',
        alpha=0.9,
        size=10,
        jitter=True,
        dodge=False,
        ax=ax
    )

    # Overlay Achilles as stars (*)
    df_achilles = df_beh[df_beh['rat'] == 'achilles']
    for _, row in df_achilles.iterrows():
        ax.scatter(
            x=label_to_pos[row['area_label']],
            y=row['SI'],
            marker='*',
            color='black',
            alpha = 0.9,
            s=250,
            linewidth=0.6,
            zorder=5
        )

    # === Independent pairwise t-tests ===
    pairs = [
        ('Overall', 'Deep'),
        ('Overall', 'Superficial'),
        ('Deep', 'Superficial')
    ]
    y_max = df_beh['SI'].max()
    increment = 0.05

    for idx, (label1, label2) in enumerate(pairs):
        vals1 = df_beh.loc[df_beh['area_label'].eq(label1), 'SI'].dropna().values
        vals2 = df_beh.loc[df_beh['area_label'].eq(label2), 'SI'].dropna().values
        stat, pval = ttest_ind(vals1, vals2, equal_var=False)

        signif = next((symbol for alpha, symbol in alpha_levels if pval < alpha), 'n.s.')
        if signif != 'n.s.':
            x1, x2 = label_to_pos[label1], label_to_pos[label2]
            y = y_max + (idx + 1) * increment
            ax.plot([x1, x1, x2, x2], [y, y + 0.01, y + 0.01, y], lw=1.3, color='black')
            ax.text((x1 + x2) / 2, y + 0.015, signif, ha='center', va='bottom', fontsize=12)

    # === Paired test Deep vs Superficial ===
    df_wide = df_beh.pivot_table(
        index=['rat', 'session', 'probe'],
        columns='area_label',
        values='SI'
    ).reset_index()

    df_paired = df_wide.dropna(subset=['Deep', 'Superficial'])

    if not df_paired.empty:
        stat, pval = ttest_rel(df_paired['Deep'], df_paired['Superficial'])
        print(f"Paired t-test Deep vs Superficial ({beh}): stat={stat:.3f}, p={pval:.3g}")

        signif = next((symbol for alpha, symbol in alpha_levels if pval < alpha), 'n.s.')
        if signif != 'n.s.':
            x1, x2 = label_to_pos['Deep'], label_to_pos['Superficial']
            y = y_max + (len(pairs) + 1) * increment
            ax.plot([x1, x1, x2, x2], [y, y + 0.01, y + 0.01, y],
                    lw=1.3, color='red')
            ax.text((x1 + x2) / 2, y + 0.015, f"Paired {signif}",
                    ha='center', va='bottom', fontsize=12, color='red')

        # === Connect paired dots with lines ===
        for _, row in df_paired.iterrows():
            ax.plot(
                [label_to_pos['Deep'], label_to_pos['Superficial']],
                [row['Deep'], row['Superficial']],
                color='gray', alpha=0.5, lw=1
            )

    # Labels and formatting
    ax.set_title(f"Structure Index (SI) for '{beh}' at Filter Time = 0.2 s")
    ax.set_ylabel('Structure Index (SI)')
    ax.set_xlabel('Region')
    ax.set_ylim(0, y_max + 0.35)
    sns.despine()
    plt.tight_layout()

    # Save
    base_name = f"SI_bar_{beh}_filter_0.2s_allstats_paired"
    png_path = os.path.join(output_dir, base_name + ".png")
    svg_path = os.path.join(output_dir, base_name + ".svg")
    fig.savefig(png_path, dpi=300, bbox_inches="tight")
    fig.savefig(svg_path, dpi=300, bbox_inches="tight")
    print(f"Saved plot for '{beh}' to:\n  - {png_path}\n  - {svg_path}")
    plt.show()
    plt.close()

#################################################################################################################
###################################################################################################################
import matplotlib.pyplot as plt
import seaborn as sns

# Pivot the data to wide format (deep & superficial as columns)
df_wide = df_filtered.pivot_table(
    index=['rat', 'session', 'probe', 'behavioral_label'],
    values=['si_deep', 'si_sup']
).reset_index()

# Make one scatter plot per behavioral variable
for beh in ['pos', 'time']:
    df_beh = df_wide[df_wide['behavioral_label'] == beh]

    fig, ax = plt.subplots(figsize=(6,6))

    # Scatterplot
    sns.scatterplot(
        data=df_beh,
        x='si_deep',
        y='si_sup',
        hue='rat',         # color per rat if useful
        style='rat',       # different marker per rat (optional)
        s=250,
        ax=ax
    )

    # Diagonal reference line
    lims = [
        min(df_beh[['si_deep','si_sup']].min()) - 0.05,
        max(df_beh[['si_deep','si_sup']].max()) + 0.05
    ]
    ax.plot(lims, lims, 'k--', alpha=0.6, lw=1)

    ax.set_xlim(lims)
    ax.set_ylim(lims)
    ax.set_xlabel('SI Deep')
    ax.set_ylabel('SI Superficial')
    ax.set_title(f"SI Deep vs SI Superficial ({beh})")

    base_name = f"SI_scatter_{beh}_filter_0.2s"
    png_path = os.path.join(output_dir, base_name + ".png")
    svg_path = os.path.join(output_dir, base_name + ".svg")
    fig.savefig(png_path, dpi=300, bbox_inches="tight")
    fig.savefig(svg_path, dpi=300, bbox_inches="tight")
    sns.despine()
    plt.tight_layout()
    plt.show()