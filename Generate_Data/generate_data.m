%% Generate data using the relation G(\tau)=K�A(\omega)
% By Romain Fournier

% parameters name of the input file containging :
%     nb_data: nb_data to be generated
%     beta : inverse temperature (1/k_bT)
%     nb_pics : maximum number of pics
%     output : prefix of the output file
%     nb_gL : number of legendre coefficients
%     nb_tau: number of intervals for tau (0,BETA)
%     nb_omega: number of intervals for omega(-WO W0)
%     omega_0 : max frequency
%     legendre_method : exact or approximate
%     noise_level : maximum noise added to coefficient G 

% output :
%     file : output_Gl_name.dat
%     file : output_Aw_name.dat
%     file : output_G_name.dat

function generate_data(input_file)
tic
%% read the parameter file
parameter_values=read_file(input_file);
%% assign the parameter and display messages if default values were used
%beta
if isfield(parameter_values,'beta')
    BETA=parameter_values.beta;
else
    warning('Parameter beta not found, beta=10 was used')
    BETA=10;
end
%nb_data
if isfield(parameter_values,'nb_data')
    NB_DATA=parameter_values.nb_data;
else
    warning('Parameter nb_data not found, nb_data=1000 was used')
    NB_DATA=1000;
end
%nb_pics
if isfield(parameter_values,'nb_pics')
    NB_PICS=parameter_values.nb_pics;
else
    warning('Parameter nb_pics not found, nb_pics=21 was used')
    NB_PICS=21;
end
%output
if isfield(parameter_values,'output')
    output=parameter_values.output;
else
    warning("Parameter output not found, output='output' was used")
    output='output';
end
%nb_gl
if isfield(parameter_values,'nb_gl')
    NB_GL=parameter_values.nb_gl;
else
    warning('Parameter nb_gl not found, nb_gl=64 was used')
    NB_GL=64;
end
%nb_omega
if isfield(parameter_values,'nb_omega')
    NB_OMEGA=parameter_values.nb_omega;
else
    warning('Parameter nb_omega not found, nb_omega=1024 was used')
    NB_OMEGA=1024;
end
%W0
if isfield(parameter_values,'omega0')
    OMEGA_0=parameter_values.nb_gl;
else
    warning('Parameter omega0 not found, omega0=15 was used')
    OMEGA_0=15;
end
%tau
if isfield(parameter_values,'nb_tau')
    NB_TAU=parameter_values.nb_tau;
else
    warning('Parameter nb_tau not found, nb_tau=1024 was used')
    NB_TAU=1024;
end
%legendre method
if isfield(parameter_values,'legendre_method')
    legendre_method=parameter_values.legendre_method;
else
    warning('Parameter legendre_method not found, legendre_method=exact was used')
    legendre_method='exact';
end
%legendre method
if isfield(parameter_values,'integral_tol')
    integral_tol=parameter_values.integral_tol;
else
    warning('Parameter integral_tol not found, legendre_method=0.0001 was used')
    integral_tol=0.0001;
end
%show example
if isfield(parameter_values,'example')
    example=parameter_values.example;
else
    warning('Parameter example not found, example=false was used')
    example=0;
end
%Noise level
if isfield(parameter_values,'noise_level')
    NOISE_LEVEL=parameter_values.noise_level;
else
    warning('Parameter noise_level not found, NOISE_LEVEL=0 was used')
    NOISE_LEVEL=0;
end

%% Generate the A
% A=sum_1:random(nbpics) gaussian(mu,sigma)
%generate the centers and the variance (there must be a small pic around
%the center

wr=rand(1,NB_DATA,NB_PICS).*12-6;
sigma=rand(1,NB_DATA,NB_PICS)*2.3+0.6;
wr(1,:,1)=rand(NB_DATA,1)-0.5;
sigma(1,:,1)=rand(NB_DATA,1)*0.9+0.1;
%how much term does each data contain
R=randi(NB_PICS,NB_DATA,1);

cancellator=ones(1,NB_DATA,NB_PICS);
for ii=1:NB_DATA
    cancellator(1,ii,R(ii)+1:end)=0;
end
%discretize the interval
omega=linspace(-OMEGA_0,OMEGA_0,NB_OMEGA);
A_sum_handlers= @(x) (sum(cancellator(1,:,:).*normpdf(x,wr(1,:,:),sigma(1,:,:)),3))';

A=A_sum_handlers(omega');
NORMALIZATION_FACTOR=trapz(omega,A,2);
A=A./NORMALIZATION_FACTOR;
%% Compute G
%G(tau)=integrate k*A dw

%one must make sure that the term do not become to big, so we devide par
%the exponential if the argument is bigger than 37
kernel_handler = @(tau,x) (x'.*(BETA-tau)>=37).*exp(-x'.*tau)./(1+exp(-x'*BETA))+(x'.*(BETA-tau)<37).*exp(x'.*(BETA-tau))./(exp(BETA*x)+1);

G_handlers= @(tau)normrnd( -integral(@(x) kernel_handler(tau,x)*A_sum_handlers(x')',-OMEGA_0,OMEGA_0,'ArrayValued',true,'AbsTol',integral_tol)./NORMALIZATION_FACTOR',NOISE_LEVEL);

taus=linspace(0,BETA,NB_TAU);
G=(G_handlers(taus')');
%G=G+(rand(size(G))-0.5)*NOISE_LEVEL;

%% Transform into legendre
%we must not forget that legendre is defined for transforming a function
%which arguement are in (0,1)
toc
switch (legendre_method)
    case 'approximate'
        legendre_x_l=zeros(NB_GL,length(taus));
        u=linspace(-1,1,length(taus));
        tensor=zeros(NB_DATA,NB_GL,NB_TAU);
        for ii=0:NB_GL-1
            legendre_x_l(ii+1,:)=legendreP(ii,u);
            tensor(:,ii+1,:)=G.*legendre_x_l(ii+1,:);
        end
        nl=trapz(u,tensor,3)/BETA;
        
    otherwise
        % Exact computation, takes time
        nl_handler=@(l) integral((@(x)  (G_handlers(x*BETA)'*legendreP(l,2*x-1))) ,0,1, 'ArrayValued',true,'AbsTol',integral_tol);
        nl= nl_handler(0:(NB_GL-1));
end

toc

%% Show the results
if(example)
    G1=@(x) sum( (2*(0:(NB_GL-1))+1).*nl(1,:).*legendreP((0:NB_GL-1),2*x/BETA-1),2);
    G_reconstruit=zeros(1,length(taus));
    for ii = 1:length(taus)
        G_reconstruit(ii)=G1(taus(ii));
    end
    figure(1)
    subplot(1,2,1)
    p=plot(omega,A(1,:));
    set(p,'linewidth',1.2);
    set(gca,'fontsize',22)
    grid on
    xlabel('\omega')
    ylabel('A(\omega)')
   
    subplot(1,2,2)
    grid on
    hold on
    p=plot(taus,G(1,:),'color','blue');
    set(p,'linewidth',1.2);
    p=plot(taus,G_reconstruit,'--','color','red');
    set(p,'linewidth',1.2);
    set(gca,'fontsize',22)
    xlabel('\tau')
    ylabel('G(\tau)')
    legend('original','recovered from legendre')
    
    p = get(gca, 'Position');
    h = axes('Parent', gcf, 'Position', [p(1)+.12 p(2)+.12 p(3)-.15 p(4)-.5]);
    b=plot(h,0:15,nl(1:16),'x-');
    set(gca,'fontsize',20)
    set(b,'linewidth',1.2);
    set(gca,'XTick',0:2:15)
    xlabel('l')
    ylabel('n_l')
    
%     dlmwrite('../Figures_factory/Figure1/Data/G.dat',G,'delimiter',',')
%     dlmwrite('../Figures_factory/Figure1/Data/tau.dat',taus,'delimiter',',')
%     dlmwrite('../Figures_factory/Figure1/Data/A.dat',A,'delimiter',',')
%     dlmwrite('../Figures_factory/Figure1/Data/omega.dat',omega,'delimiter',',')
%     dlmwrite('../Figures_factory/Figure1/Data/nl.dat',nl,'delimiter',',')
end
%% Save the data
toc
dlmwrite(strcat('../Database/G_',output,'.csv'),G,'-append','delimiter',',')
dlmwrite(strcat('../Database/nl_',output,'.csv'),nl,'-append','delimiter',',')
dlmwrite(strcat('../Database/A_',output,'.csv'),A,'-append','delimiter',',')
%For maxent, the convention is different.
dlmwrite(strcat('../Database/Spect_in'),[omega(:),A(1,:)'],'-append','delimiter',' ')
for ii=1:length(nl(:,1))
    dlmwrite(strcat('../Database/Gl',output,'.dat'),[(0:NB_GL-1)',nl(ii,:)'.*sqrt(2*(0:NB_GL-1)'+1)*BETA,repmat(2e-9,NB_GL,1)],'-append','delimiter',' ')
end

end
%% function read_file (thanks to mathworks)
% parameter :
% input_file : name of the file to read

function map=read_file(input_file)
%% Populate parameters struct from parameters.txt
fid = fopen(input_file, 'r');
if fid < 0; error('Parameters file is missing. Please replace.'); end
% Extract key-value pairs from params file
keys = {}; values = {};
while ~feof(fid)
    this_line = fgets(fid);
    % Check to see if the line is a comment or whitespace
    switch this_line(1)
        case {'#', ' ', char(10)}
        otherwise
            % First token is key; second is value
            [keys{end+1} inds] = strtok(this_line, '=');
            values{end+1} = strtok(inds, '=');
            % If the value is convertible to num, convert to num
            if ~isnan(str2double(values{end}))
                values{end} = str2double(values{end});
            end
    end
end
fclose(fid);
% Remove extra padding from key-value pairs
keys = strtrim(keys);
str_ind = cellfun(@ischar, values);
values(str_ind) = strtrim(values(str_ind));
% Remove extra ' ' around strings
data_type_ind = [];
for i = find(str_ind)
    str = values{i};
    if str(1) == ''''
        values{i} = str(2:end-1);
    else
        data_type_ind(end+1) = i;
    end
end
% Convert data type assignments to actual data types
% This may actually be a valid use of the eval function!
for i = data_type_ind
    eval(['values{i} = ' values{i} ';']);
end
% Construct parameter struct from read key-value pairs
map = cell2struct(values, keys, 2);
end

