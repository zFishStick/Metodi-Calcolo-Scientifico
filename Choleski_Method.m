

folder = "C:\Users\Diagon\Desktop\UNIMIB\ANNO 1\SECONDO SEMESTRE\Metodi Calcolo\Matrici-Sparse";
matrix_names = {'ex15.mat', 'apache2.mat', 'cfd1.mat', 'cfd2.mat', 'Flan_1565.mat', 'G3_circuit.mat', 'parabolic_fem.mat', 'shallow_water1.mat', 'StocF-1465.mat'};


nomi = {};
dimensioni = [];
tempi = [];
errori = [];
memorie = [];

for i = 1 : length(matrix_names)

    data = load(fullfile(folder, matrix_names{i}));

    A = data.Problem.A;
    
    if not(issymmetric(A))
        disp("La matrice non è simmetrica");
    end
    
    disp(['Nome matrice: ' , matrix_names{i}])
    disp([ 'Dimensione matrice : ' ,num2str(size(A,1)), 'x' ,num2str(size(A,2))]);
    
    xe = ones(length(A), 1); % Vettore con tutti 1
    
    mem_prima = whos('A');
    b = A * xe;
    
    % Tempo
    tic
    x = A \ b;
    t = toc;
    
    % Errore
    rel_err = norm(x - xe)/norm(xe);
    
    mem_dopo = whos('x');
    mem_MB = (mem_prima.bytes + mem_dopo.bytes) / 1024^2;

    nomi{end+1} = matrix_names{i};
    dimensioni(end+1) = size(A,1);
    tempi(end+1) = t;
    errori(end+1) = rel_err;
    memorie(end+1) = mem_MB;
end

if ispc
    risultati = "risultati_win_matlab.csv";
else
    risultati = "risultati_linux_matlab.csv";
end

T = table(nomi', dimensioni', tempi', errori', memorie', ...
    'VariableNames', {'nome','dimensione','tempo','errore','memoria'});

T = sortrows(T, 'dimensione');

writetable(T, risultati);
