from  .loader import load_new_data, load_citation_data, load_data_amazon, load_data_flickr, load_data_ogbn
from config import parser
args = parser.parse_args()

def load_data(args):

    if args.dataset_name in ['squirrel', 'chameleon', 'actor']:            
        return load_new_data(args)

    elif args.dataset_name in ['cora', 'citeseer', 'pubmed']:
        return load_citation_data(args)
    
    elif args.dataset_name == 'amazon':
        return load_data_amazon(args)

    elif args.dataset_name == 'flickr':
        return load_data_flickr(args)
    
    elif args.dataset_name in ['ogbn-arxiv']:
        return load_data_ogbn(args)
    