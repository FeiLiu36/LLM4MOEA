function Offspring = OperatorLO(Problem,Parent1,Parent2,Parent3,ParentSet)
%OperatorDE - The operator of differential evolution.
%
%   Off = OperatorDE(Pro,P1,P2,P3) uses the operator of differential
%   evolution to generate offsprings for problem Pro based on parents P1,
%   P2, and P3. If P1, P2, and P3 are arrays of SOLUTION objects, then Off
%   is also an array of SOLUTION objects; while if P1, P2, and P3 are
%   matrices of decision variables, then Off is also a matrix of decision
%   variables, i.e., the offsprings are not evaluated. Each object or row
%   of P1, P2, and P3 is used to generate one offspring by P1 + 0.5*(P2-P3)
%   and polynomial mutation.
%
%	Off = OperatorDE(Pro,P1,P2,P3,{CR,F,proM,disM}) specifies the
%	parameters of operators, where CR and F are the parameters in
%	differental evolution, proM is the expectation of the number of mutated
%	variables, and disM is the distribution index of polynomial mutation.
%
%   Example:
%       Off = OperatorDE(Problem,Parent1,Parent2,Parent3)
%       Off = OperatorDE(Problem,Parent1.decs,Parent2.decs,Parent3.decs,{1,0.5,1,20})

%------------------------------- Reference --------------------------------
% H. Li and Q. Zhang, Multiobjective optimization problems with complicated
% Pareto sets, MOEA/D and NSGA-II, IEEE Transactions on Evolutionary
% Computation, 2009, 13(2): 284-302.
%------------------------------- Copyright --------------------------------
% Copyright (c) 2023 BIMK Group. You are free to use the PlatEMO for
% research purposes. All publications which use this platform or any code
% in the platform should acknowledge the use of "PlatEMO" and reference "Ye
% Tian, Ran Cheng, Xingyi Zhang, and Yaochu Jin, PlatEMO: A MATLAB platform
% for evolutionary multi-objective optimization [educational forum], IEEE
% Computational Intelligence Magazine, 2017, 12(4): 73-87".
%--------------------------------------------------------------------------

    %% Parameter setting

    [CR,F,proM,disM] = deal(1,0.5,1,20);

    
    if isa(Parent1(1),'SOLUTION')
        evaluated = true;
        Parent1   = Parent1.decs;
        Parent2   = Parent2.decs;
        Parent3   = Parent3.decs;
    else
        evaluated = false;
    end
    [N,D] = size(Parent1);
    
    % Initialize weights vector
    L = size(ParentSet);
    L = L(2);
    LLMw = zeros(1,L);

    % Calculate softmax for each rank value
    for i = 1:L
        r_i = (L+1-i)/L;
        weight = -0.111 * r_i^3 + 1.037 * r_i^2 - 1.291 * r_i + 0.445;
        LLMw(i) = weight;  
    end
    
    % Normalize LLMw vector
    LLMw = LLMw / sum(LLMw);
    
    LLMw = LLMw + 0.5*randn(1,L);

    Offspring = Parent1;
    OffspringLLM = 0;
    for i = 1:length(LLMw)
        OffspringLLM = OffspringLLM + LLMw(i) * ParentSet(i).decs;
    end
    
    Site = rand(N,D) < 0.1;

    Offspring(Site) = OffspringLLM(Site);
    %scale = 0.9 + (1.1 - 0.9) * rand(1, numel(Offspring));
    %numel(Offspring)
    %Offspring = Offspring .* scale;


    
    %% Polynomial mutation
    Lower = repmat(Problem.lower,N,1);
    Upper = repmat(Problem.upper,N,1);
    Site  = rand(N,D) < proM/D;
    mu    = rand(N,D);
    temp  = Site & mu<=0.5;
    Offspring       = min(max(Offspring,Lower),Upper);
    Offspring(temp) = Offspring(temp)+(Upper(temp)-Lower(temp)).*((2.*mu(temp)+(1-2.*mu(temp)).*...
                      (1-(Offspring(temp)-Lower(temp))./(Upper(temp)-Lower(temp))).^(disM+1)).^(1/(disM+1))-1);
    temp = Site & mu>0.5; 
    Offspring(temp) = Offspring(temp)+(Upper(temp)-Lower(temp)).*(1-(2.*(1-mu(temp))+2.*(mu(temp)-0.5).*...
                      (1-(Upper(temp)-Offspring(temp))./(Upper(temp)-Lower(temp))).^(disM+1)).^(1/(disM+1)));
    if evaluated
        Offspring = Problem.Evaluation(Offspring);
    end
end