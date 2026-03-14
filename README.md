plots_drifts_strips.py
- main script used for IC mode and readout strip mode
- channel-specific data extraction
- creates plots and different .csv files

plot_integrated_charge_signals.py
- used to plot integrated signal against cathode/mesh voltage
- uses signal_mean_summary.csv file from plots_drifts_strips.py -> manually renamed to signal_mean_summaryXXX.csv, where XXX denotes the cathode/mesh voltage, rounded
- dictionary (12-21) stores exact voltage numbers
  
