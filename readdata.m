%% Read in the h5 file from PFLOTRAN output and plot and process data

cd('C:\Users\yz60069\TAI\TAI_fresh')
h5disp('TAI_wetland2.h5')   % display the content
set(0,'DefaultFigureWindowStyle','docked')

%% specify the PFLOTRAN setup
nx = 10;
ny = 10;
nz = 9;
ngrids = nx * ny * nz;
ntpt = 16;   %number of timepoints

Vars = {'Total_O2(aq) [M]','Total_CH4(aq) [M]','Total_DOM1 [M]','Total_SO4-- [M]','Total_H2S(aq) [M]','Liquid_Saturation'};  %variables by PFLOTRAN output, use number to call the variable 
varNames = {'o2','ch4','dom','so4','h2s','liquid'};
varTypes = {'cell','cell','cell','cell','cell', 'cell'};
Conc = table('Size',[ntpt 6],'VariableTypes',varTypes,'VariableNames',varNames);


%% read in data

% Concentration data
for var_id = 1:6
    for i = 1:ntpt
        timepoint = i - 1;
        ds = ['/Time:  ',sprintf('%.5E',timepoint),' d/', char(Vars(var_id))];   %create the string for dataset name
        data_t = h5read('TAI_wetland2.h5', ds);

        Conc{i,var_id} = {data_t};
    end
end


% depth data
depth = [];


%% plot heatmap of concentration
timepoint = 16;
data = Conc.o2{timepoint};

rootlayer = 4;
heatmap(flipud(squeeze(data(rootlayer, :, :))));


%% plot the depth profiles

        

    


