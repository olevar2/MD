import React from 'react';
import { 
  DataGrid, 
  DataGridProps, 
  GridColDef, 
  GridToolbar,
  GridSelectionModel
} from '@mui/x-data-grid';
import { Box, styled } from '@mui/material';

export interface DataTableProps extends Omit<DataGridProps, 'columns'> {
  columns: GridColDef[];
  data: any[];
  loading?: boolean;
  pageSize?: number;
  selectable?: boolean;
  onSelectionChange?: (selectionModel: GridSelectionModel) => void;
  toolbarEnabled?: boolean;
  height?: string | number;
  stickyHeader?: boolean;
}

const StyledDataGrid = styled(DataGrid)(({ theme }) => ({
  border: 'none',
  '& .MuiDataGrid-cell:focus': {
    outline: 'none',
  },
  '& .MuiDataGrid-columnHeaders': {
    backgroundColor: theme.palette.background.default,
    borderBottom: `1px solid ${theme.palette.divider}`,
  },
  '& .MuiDataGrid-virtualScroller': {
    backgroundColor: theme.palette.background.paper,
  },
  '& .MuiDataGrid-footerContainer': {
    borderTop: `1px solid ${theme.palette.divider}`,
    backgroundColor: theme.palette.background.default,
  },
  '& .MuiDataGrid-row:hover': {
    backgroundColor: `${theme.palette.action.hover} !important`,
  },
}));

export const DataTable: React.FC<DataTableProps> = ({
  columns,
  data,
  loading = false,
  pageSize = 10,
  selectable = false,
  onSelectionChange,
  toolbarEnabled = false,
  height = 400,
  stickyHeader = false,
  ...props
}) => {
  const [paginationModel, setPaginationModel] = React.useState({
    pageSize: pageSize,
    page: 0,
  });

  return (
    <Box sx={{ width: '100%', height }}>
      <StyledDataGrid
        rows={data}
        columns={columns}
        loading={loading}
        disableColumnMenu
        disableRowSelectionOnClick={!selectable}
        checkboxSelection={selectable}
        onRowSelectionModelChange={onSelectionChange}
        pagination
        paginationModel={paginationModel}
        onPaginationModelChange={setPaginationModel}
        pageSizeOptions={[5, 10, 25, 50, 100]}
        slots={{
          toolbar: toolbarEnabled ? GridToolbar : undefined,
        }}
        sx={{
          '& .MuiDataGrid-columnHeaders': {
            position: stickyHeader ? 'sticky' : 'static',
            top: 0,
            zIndex: 2,
          }
        }}
        {...props}
      />
    </Box>
  );
};

export default DataTable;
