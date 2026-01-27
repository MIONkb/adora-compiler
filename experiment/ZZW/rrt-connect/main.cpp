#include <iostream>
#include <random>
#include <math.h>

const int kOBSTACLE_NUM = 2;
const int kSAMPLE_NUM = 4; // samples per iteration
const int kMAX_ITER = 200;
const int kMAX_NODE_NUM = kMAX_ITER * kSAMPLE_NUM + 10;
using dtype = float;
using std::cout;
using std::endl;

void sampler(dtype &x, dtype &y)
{
    // constrain map bounds
    const dtype lb_x = -2, rb_x = 12, lb_y = -2, rb_y = 12;
    x = lb_x + ((dtype)rand() / (dtype)RAND_MAX) * (rb_x - lb_x);
    y = lb_y + ((dtype)rand() / (dtype)RAND_MAX) * (rb_y - lb_y);
}

inline dtype l2_norm(dtype x1, dtype y1, dtype x2, dtype y2)
{
    dtype x = x2 - x1;
    dtype y = y2 - y1;
    dtype l = sqrtf32(x * x + y * y);
    return l;
}

inline dtype l1_norm(dtype x1, dtype y1, dtype x2, dtype y2)
{
    dtype x = x2 - x1;
    dtype y = y2 - y1;
    dtype l = fabsf32(x) + fabsf32(y);
    return l;
}

inline dtype cross_product(dtype x1, dtype y1, dtype x2, dtype y2, dtype x3, dtype y3)
{
    return (x2 - x1) * (y3 - y1) - (y2 - y1) * (x3 - x1);
}

// (p1p2)(p3p4) are two line segments; return True when they intersect
bool Segment_Overlap_Checker(dtype p1_x, dtype p1_y, dtype p2_x, dtype p2_y,
                             dtype p3_x, dtype p3_y, dtype p4_x, dtype p4_y)
{
    dtype d1 = cross_product(p3_x, p3_y, p4_x, p4_y, p1_x, p1_y); // P1 relative to segment P3P4
    dtype d2 = cross_product(p3_x, p3_y, p4_x, p4_y, p2_x, p2_y); // P2 relative to segment P3P4
    dtype d3 = cross_product(p1_x, p1_y, p2_x, p2_y, p3_x, p3_y); // P3 relative to segment P1P2
    dtype d4 = cross_product(p1_x, p1_y, p2_x, p2_y, p4_x, p4_y); // P4 relative to segment P1P2

    // check strict intersection
    if (((d1 > 0.0 && d2 < 0.0) || (d1 < 0.0 && d2 > 0.0)) &&
        ((d3 > 0.0 && d4 < 0.0) || (d3 < 0.0 && d4 > 0.0)))
    {
        return true;
    }

    return false;
}

// result==0 means no conflict
bool Collision_Checker(dtype *obs_x1, dtype *obs_x2, dtype *obs_y1, dtype *obs_y2,
                       dtype p_x1, dtype p_x2, dtype p_y1, dtype p_y2)
{
    for (int i = 0; i < kOBSTACLE_NUM; i++)
    {
        bool rt = Segment_Overlap_Checker(
            obs_x1[i], obs_y1[i], obs_x2[i], obs_y2[i],
            p_x1, p_y1, p_x2, p_y2);
        if (rt)
        {
            return true;
        }
    }
    return false;
}

void Get_Nearest_Node(dtype x, dtype y, dtype *tree_x, dtype *tree_y, int tree_num, int &node, dtype &l)
{
    const dtype inf = 1000000;
    dtype tmp_l = inf;
    int tmp_node = 0;
    for (int i = 0; i < tree_num; i++)
    {
        dtype l = l1_norm(x, y, tree_x[i], tree_y[i]);
        if (l < tmp_l)
        {
            tmp_l = l;
            tmp_node = i;
        }
    }
    l = tmp_l;
    node = tmp_node;
}

void Aux_Kernel_Get_Nearest_Node(dtype *sample_x, dtype *sample_y,
                                 dtype *tree_x, dtype *tree_y, int tree_num,
                                 int *nearest_node, dtype *nearest_length)
{
    for (int j = 0; j < kSAMPLE_NUM; j++)
        Get_Nearest_Node(sample_x[j], sample_y[j], tree_x, tree_y, tree_num, nearest_node[j], nearest_length[j]);
}

void Aux_Kernel_Collision_Checker(dtype *obs_x1, dtype *obs_x2, dtype *obs_y1, dtype *obs_y2,
                                  dtype *p_x1, dtype *p_x2, dtype *p_y1, dtype *p_y2, bool *result)
{
    for (int i = 0; i < kSAMPLE_NUM; i++)
    {
        result[i] = Collision_Checker(
            obs_x1, obs_x2, obs_y1, obs_y2,
            p_x1[i], p_x2[i], p_y1[i], p_y2[i]);
    }
}

// update the corresponding tree based on the sample point
void Update_Tree(dtype *obs_x1, dtype *obs_x2, dtype *obs_y1, dtype *obs_y2,
                 dtype *sample_x, dtype *sample_y, bool *result, dtype step,
                 dtype *tree_x, dtype *tree_y, int *f_tree, int &tree_num)
{
    // used to store the nearest node
    int *nearest_node = new int[kSAMPLE_NUM];
    dtype *nearest_length = new dtype[kSAMPLE_NUM];
    dtype *t_x = new dtype[kSAMPLE_NUM];
    dtype *t_y = new dtype[kSAMPLE_NUM];

    // find the nearest point on the tree
    Aux_Kernel_Get_Nearest_Node(sample_x, sample_y,
                                tree_x, tree_y, tree_num,
                                nearest_node, nearest_length);

    // adjust length by step; this modifies sample as intended
    for (int j = 0; j < kSAMPLE_NUM; j++)
    {
        int i = nearest_node[j];
        if (nearest_length[j] > step)
        {
            dtype ratio = step / nearest_length[j];
            sample_x[j] = tree_x[i] + (sample_x[j] - tree_x[i]) * ratio;
            sample_y[j] = tree_y[i] + (sample_y[j] - tree_y[i]) * ratio;
        }
        t_x[j] = tree_x[i];
        t_y[j] = tree_y[i];
    }

    // result=1 means there is a conflict
    Aux_Kernel_Collision_Checker(obs_x1, obs_x2, obs_y1, obs_y2,
                                 sample_x, t_x, sample_y, t_y, result);

    // update the tree if there is no conflict
    for (int i = 0; i < kSAMPLE_NUM; i++)
    {
        if (result[i])
            continue;
        f_tree[tree_num] = nearest_node[i];
        tree_x[tree_num] = sample_x[i];
        tree_y[tree_num] = sample_y[i];
        ++tree_num;
    }

    delete[] nearest_node;
    delete[] nearest_length;
    delete[] t_x;
    delete[] t_y;
}

// check whether the sample connects directly to the tree; otherwise update the tree
bool Has_Connect(dtype *obs_x1, dtype *obs_x2, dtype *obs_y1, dtype *obs_y2,
                 dtype *sample_x, dtype *sample_y, bool *result_sample, dtype step,
                 dtype *tree_x, dtype *tree_y, int *f_tree, int &tree_num,
                 int &count, int &tree_node)
{
    bool *tmp_result_update = new bool[kSAMPLE_NUM];
    bool *tmp_result_near = new bool[kSAMPLE_NUM];
    dtype *tmp_sample_x = new dtype[kSAMPLE_NUM];
    dtype *tmp_sample_y = new dtype[kSAMPLE_NUM];

    // used to store the nearest node
    int *nearest_node = new int[kSAMPLE_NUM];
    dtype *nearest_length = new dtype[kSAMPLE_NUM];
    dtype *t_x = new dtype[kSAMPLE_NUM];
    dtype *t_y = new dtype[kSAMPLE_NUM];

    // find the nearest point on the tree
    Aux_Kernel_Get_Nearest_Node(sample_x, sample_y,
                                tree_x, tree_y, tree_num,
                                nearest_node, nearest_length);

    // adjust length by step
    for (int j = 0; j < kSAMPLE_NUM; j++)
    {
        int i = nearest_node[j];
        if (nearest_length[j] > step)
        {
            dtype ratio = step / nearest_length[j];
            tmp_sample_x[j] = tree_x[i] + (sample_x[j] - tree_x[i]) * ratio;
            tmp_sample_y[j] = tree_y[i] + (sample_y[j] - tree_y[i]) * ratio;
            tmp_result_near[j] = 0;
        }
        else
            tmp_result_near[j] = 1;
        t_x[j] = tree_x[i];
        t_y[j] = tree_y[i];
    }

    // result=1 means there is a conflict
    Aux_Kernel_Collision_Checker(obs_x1, obs_x2, obs_y1, obs_y2,
                                 tmp_sample_x, t_x, tmp_sample_y, t_y, tmp_result_update);

    bool ret = false;
    int new_count = 0;
    for (int i = 0; i < kSAMPLE_NUM; i++)
    {
        if (result_sample[i] == 0)
            new_count++;
        if (tmp_result_update[i] != 0)
            continue;
        if (tmp_result_near[i] && !result_sample[i])
        {
            // connect directly
            count = new_count - 1;
            tree_node = nearest_node[i];
            ret = true;
            break;
        }
        else
        {
            f_tree[tree_num] = nearest_node[i];
            tree_x[tree_num] = tmp_sample_x[i];
            tree_y[tree_num] = tmp_sample_y[i];
            ++tree_num;
        }
    }

    delete[] tmp_result_update;
    delete[] tmp_result_near;
    delete[] tmp_sample_x;
    delete[] tmp_sample_y;

    delete[] nearest_node;
    delete[] nearest_length;
    delete[] t_x;
    delete[] t_y;

    return ret;
}

void Get_Answer_Array(dtype *&answer_x, dtype *&answer_y, int &answer_num,
                      dtype *src_tree_x, dtype *src_tree_y, dtype *dst_tree_x, dtype *dst_tree_y,
                      int *f_src_tree, int *f_dst_tree,
                      int leaf_src, int leaf_dst)
{
    int *aux_node_stack = new int[kMAX_NODE_NUM];
    int aux_num = 0;
    int node = leaf_src;
    while (node != 0)
    {
        aux_node_stack[aux_num++] = node;
        node = f_src_tree[node];
    }
    answer_num = 1;
    answer_x[0] = src_tree_x[0];
    answer_y[0] = src_tree_y[0];
    for (int i = aux_num - 1; i >= 0; i--)
    {
        int node = aux_node_stack[i];
        answer_x[answer_num] = src_tree_x[node];
        answer_y[answer_num] = src_tree_y[node];
        answer_num++;
    }
    node = leaf_dst;
    while (node != 0)
    {
        answer_x[answer_num] = dst_tree_x[node];
        answer_y[answer_num] = dst_tree_y[node];
        answer_num++;
        node = f_dst_tree[node];
    }
    answer_x[answer_num] = dst_tree_x[0];
    answer_y[answer_num] = dst_tree_y[0];
    answer_num++;
    delete[] aux_node_stack;
}

// return 1, path found successfully
bool RRT_Connect_Search(dtype src_x, dtype src_y, dtype dst_x, dtype dst_y,
                        dtype *obs_x1, dtype *obs_x2, dtype *obs_y1, dtype *obs_y2,
                        dtype *&answer_x, dtype *&answer_y, int &answer_num)
{
    dtype step = 1; // can be switched to a dynamic step size

    bool ret = false;
    int leaf_src, leaf_dst;

    // store the tree
    dtype *src_tree_x = new dtype[kMAX_NODE_NUM];
    dtype *src_tree_y = new dtype[kMAX_NODE_NUM];
    dtype *dst_tree_x = new dtype[kMAX_NODE_NUM];
    dtype *dst_tree_y = new dtype[kMAX_NODE_NUM];

    int *f_src_tree = new int[kMAX_NODE_NUM];
    int *f_dst_tree = new int[kMAX_NODE_NUM];
    int src_tree_num = 1;
    int dst_tree_num = 1;

    src_tree_x[0] = src_x;
    src_tree_y[0] = src_y;
    dst_tree_x[0] = dst_x;
    dst_tree_y[0] = dst_y;
    f_src_tree[0] = 0;
    f_dst_tree[0] = 0;

    dtype *sample_x = new dtype[kSAMPLE_NUM];
    dtype *sample_y = new dtype[kSAMPLE_NUM];
    bool *sample_result = new bool[kSAMPLE_NUM];
    dtype *new_dst_x = new dtype[kSAMPLE_NUM];
    dtype *new_dst_y = new dtype[kSAMPLE_NUM];

    int iter;
    for (iter = 0; iter < kMAX_ITER; iter++)
    {
        // random sampling in space
        for (int i = 0; i < kSAMPLE_NUM; i++)
        {
            sampler(sample_x[i], sample_y[i]);
        }
        if (src_tree_num <= dst_tree_num)
        {
            int pre_tree_num = src_tree_num;
            // expand src_tree
            Update_Tree(obs_x1, obs_x2, obs_y1, obs_y2,
                        sample_x, sample_y, sample_result, step,
                        src_tree_x, src_tree_y, f_src_tree, src_tree_num);
            int count = 0, node = 0;
            bool rt1 = Has_Connect(obs_x1, obs_x2, obs_y1, obs_y2,
                                   sample_x, sample_y, sample_result, step,
                                   dst_tree_x, dst_tree_y, f_dst_tree, dst_tree_num,
                                   count, node);
            if (rt1)
            {
                ret = true;
                leaf_src = pre_tree_num + count;
                leaf_dst = node;
            }
        }
        else
        {
            int pre_tree_num = dst_tree_num;
            // expand dst_tree
            Update_Tree(obs_x1, obs_x2, obs_y1, obs_y2,
                        sample_x, sample_y, sample_result, step,
                        dst_tree_x, dst_tree_y, f_dst_tree, dst_tree_num);
            int count = 0, node = 0;
            bool rt1 = Has_Connect(obs_x1, obs_x2, obs_y1, obs_y2,
                                   sample_x, sample_y, sample_result, step,
                                   src_tree_x, src_tree_y, f_src_tree, src_tree_num,
                                   count, node);
            if (rt1)
            {
                ret = true;
                leaf_src = node;
                leaf_dst = pre_tree_num + count;
            }
        }

        if (ret)
        {
            Get_Answer_Array(answer_x, answer_y, answer_num,
                             src_tree_x, src_tree_y, dst_tree_x, dst_tree_y,
                             f_src_tree, f_dst_tree,
                             leaf_src, leaf_dst);
            break;
        }
    }

    printf("iter %d  src %d   dst %d\n", iter, src_tree_num, dst_tree_num);
    // printf("src   tree\n");
    // for (int i = 0; i < src_tree_num; i++)
    //     printf("%10.3f%10.3f\n", src_tree_x[i], src_tree_y[i]);
    // printf("dst  tree\n");
    // for (int i = 0; i < dst_tree_num; i++)
    //     printf("%10.3f%10.3f\n", dst_tree_x[i], dst_tree_y[i]);

    delete[] src_tree_x;
    delete[] src_tree_y;
    delete[] dst_tree_x;
    delete[] dst_tree_y;

    delete[] f_src_tree;
    delete[] f_dst_tree;

    delete[] sample_x;
    delete[] sample_y;
    delete[] sample_result;
    delete[] new_dst_x;
    delete[] new_dst_y;

    return ret;
}

int main()
{
    dtype *obs_x1 = new dtype[kOBSTACLE_NUM];
    dtype *obs_x2 = new dtype[kOBSTACLE_NUM];
    dtype *obs_y1 = new dtype[kOBSTACLE_NUM];
    dtype *obs_y2 = new dtype[kOBSTACLE_NUM];
    // a pair (obs_x1[n],obs_y1[n]) and (obs_x2[n],obs_y2[n]) represents a line segment
    // for (int i = 0; i < kOBSTACLE_NUM; i++)
    // {
    //     obs_x1[i] = 5;
    //     obs_y1[i] = 0;
    //     obs_x2[i] = 5;
    //     obs_y2[i] = 10;
    // }

    obs_x1[0] = 3;
    obs_y1[0] = -2;
    obs_x2[0] = 3;
    obs_y2[0] = 9;
    obs_x1[1] = 7;
    obs_y1[1] = 12;
    obs_x2[1] = 7;
    obs_y2[1] = 1;

    dtype src_x = 0, src_y = 0, dst_x = 10, dst_y = 5;
    dtype *answer_x = new dtype[1000];
    dtype *answer_y = new dtype[1000];
    int answer_num;
    bool rt = RRT_Connect_Search(src_x, src_y, dst_x, dst_y,
                                 obs_x1, obs_x2, obs_y1, obs_y2,
                                 answer_x, answer_y, answer_num);
    if (rt)
    {
        cout << "find path" << endl;
        for (int i = 0; i < answer_num; i++)
            printf("%3d :  %10.3f%10.3f\n", i, answer_x[i], answer_y[i]);
    }
    else
    {
        cout << "not find path" << endl;
    }
}