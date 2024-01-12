from  .GNN import GNN_Euclidean, GNN_LORENTZ, GNN_POINCARE
from  .GNN_mp import GNN_Euclidean_mp, GNN_LORENTZ_mp, GNN_POINCARE_mp
# from config import parser
# args = parser.parse_args()

def build_network(args):
    """Builds the neural network."""

    implemented_networks = ('GNN', 'GNN_mp')
    assert args.net_name in implemented_networks

    net = None
            
    if args.net_name == 'GNN':
        if args.manifold == 'Euclidean':
            print('Current manifold is Euclidean.')
            net = GNN_Euclidean(dropout = args.dropout, act=args.act, use_bias = args.bias, layer_dims=args.layer_dims)
        if args.manifold == 'Lorentz':
            print('Current manifold is Lorentz.')
            net = GNN_LORENTZ(manifold = args.manifold, use_bias = args.bias, dropout = args.dropout, layer_dims=args.layer_dims)
        if args.manifold == 'PoincareBall':
            print('Current manifold is Poincare.')
            net = GNN_POINCARE(manifold = args.manifold, c = args.c, dropout = args.dropout, act=args.act, use_bias = args.bias, layer_dims=args.layer_dims)
            
    if args.net_name == 'GNN_mp':
        if args.manifold == 'Euclidean':
            print('Current manifold is Euclidean.')
            net = GNN_Euclidean_mp(dropout = args.dropout, act=args.act, use_bias = args.bias, layer_dims=args.layer_dims)
        if args.manifold == 'Lorentz':
            print('Current manifold is Lorentz.')
            net = GNN_LORENTZ_mp(manifold = args.manifold, use_bias = args.bias, dropout = args.dropout, layer_dims=args.layer_dims)
        if args.manifold == 'PoincareBall':
            print('Current manifold is Poincare.')
            net = GNN_POINCARE_mp(manifold = args.manifold, c = args.c, dropout = args.dropout, act=args.act, use_bias = args.bias, layer_dims=args.layer_dims)
            


    return net
