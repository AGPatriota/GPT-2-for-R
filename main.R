require(tok)
require(torch)



if(!file.exists("Model-weights.pt")){
cat('
Make sure your have downloaded the weights at: https://drive.google.com/file/d/1jnYn3kaVyoLcGEmDBCOFNPAslrs2tRo4

Download it to the main folder.
')
} else {

source('R/GPT.R')
source('config.R')
source('R/Generators.R')

cat('If you want to generate Tokens, type Generate() \n')
}
