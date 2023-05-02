require(tok)
require(torch)



if(!file.exists("Model-weights.pt")){
cat('
Make sure you have downloaded the required weights!  
')
} else {

source('R/GPT.R')
source('config.R')
source('R/Generators.R')

cat('If you want to generate Tokens, type Generate() \n')
}
