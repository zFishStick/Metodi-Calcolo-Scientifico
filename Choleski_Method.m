
folder = "matrici";
matrix_names = {'ex15.mat', 'shallow_water1.mat', 'Flan_1565.mat'};
url = 'https://suitesparse-collection-website.herokuapp.com/mat/MaxPlanck/shallow_water1.mat';
filename = 'matrici/shallow_water1.mat';

websave(filename, url)

for i = 1 : length(matrix_names)

    data = load(strcat(folder + '/' + matrix_names{i}));

    A = data.Problem.A; % Matrice
    
    if not(issymmetric(A))
        print("La matrice non è simmetrica");
    end
    
    disp(['Nome matrice: ' , matrix_names{i}])
    disp([ 'Dimensione matrice : ' ,num2str(size(A,1)), 'x' ,num2str(size(A,2))]);
    
    xe = ones(length(A), 1); % Vettore con tutti 1
    
    b = A * xe;
    
    tic
    x = A \ b;
    toc
    
    rel_err = norm(x - xe)/norm(xe);

end

