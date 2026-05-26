for ((x=8; x<=131072; x*=2)); do
    [[ -f param_sweep_"$x"_out_with_cost_gpu.csv ]] && python plot_cube.py param_sweep_"$x"_out_with_cost_gpu.csv
    [[ -f param_sweep_"$x"_out_with_cost_cpu.csv ]] && python plot_cube.py param_sweep_"$x"_out_with_cost_cpu.csv
done
