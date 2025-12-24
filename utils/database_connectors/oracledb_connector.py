# From https://github.com/OpenDCAI/DataFlow
# Based on: 
# https://github.com/OpenDCAI/DataFlow/blob/main/dataflow/utils/text2sql/database_connector/sqlite_connector.py
# Apache License 2.0 - https://github.com/OpenDCAI/DataFlow?tab=Apache-2.0-1-ov-file#readme
# 
#  
# @article{liang2025dataflow,
#  title={DataFlow: An LLM-Driven Framework for Unified Data Preparation and Workflow Automation in the Era of Data-Centric AI},
#  author={Liang, Hao and Ma, Xiaochen and Liu, Zhou and Wong, Zhen Hao and Zhao, Zhengyang and Meng, Zimo and He, Runming and Shen, Chengyu and Cai, Qifeng and Han, Zhaoyang and others},
#  journal={arXiv preprint arXiv:2512.16676},
#  year={2025}
# }
from typing import Dict, Any, Optional, Tuple, List, NamedTuple
from abc import ABC, abstractmethod
from dataflow import get_logger
import oracledb
import time
from oracledb import LOB

class QueryResult(NamedTuple):
    success: bool
    data: List[Dict[str, Any]]
    columns: List[str]
    row_count: int
    error: Optional[str] = None

class DatabaseInfo(NamedTuple):
    db_id: str
    db_type: str
    connection_info: Dict[str, Any]
    metadata: Dict[str, Any]

class DatabaseConnectorABC(ABC):
    @abstractmethod
    def connect(self, connection_info: Dict[str, Any]) -> Any:
        pass

    @abstractmethod
    def execute_query(self, connection: Any, sql: str, params: Optional[tuple] = None) -> QueryResult:
        pass

    @abstractmethod
    def get_schema_info(self, connection: Any) -> Dict[str, Any]:
        pass

    @abstractmethod
    def discover_databases(self, config: Dict[str, Any]) -> Dict[str, DatabaseInfo]:
        pass

    @abstractmethod
    def get_number_of_special_column(self, connection: Any) -> int:
        pass

class OracleConnector(DatabaseConnectorABC):
    """Oracle database connector implementation with full schema support"""

    def __init__(self):
        self.logger = get_logger()

    def close(self,conn:oracledb.Connection):
        conn.close()

    def connect(self, connection_info: Dict) -> oracledb.Connection:
        """Connect to Oracle database"""
        config = connection_info.copy()
        # Remove encoding if present, as oracledb.connect doesn't accept it
        config.pop('encoding', None)

        try:
            # Build DSN if not provided
            if 'dsn' not in config:
                host = config.get('host', 'localhost')
                port = config.get('port', 1521)
                service_name = config.get('service_name', config.get('database', 'ORCL'))
                config['dsn'] = f"{host}:{port}/{service_name}"

            conn = oracledb.connect(**config)
            # Oracle doesn't have a simple read-only mode like MySQL
            # We'll enforce read-only through query validation
            
            return conn
        except oracledb.Error as e:
            self.logger.error(f"Oracle connection failed: {e}")
            raise

    def execute_query(self, connection: oracledb.Connection, sql: str, params: Optional[Tuple] = None) -> QueryResult:
        """Execute query with enhanced error handling and result processing"""
        start_time = time.time()
        cursor = None

        #Clean SQL
        if sql.endswith(';'):
            sql=sql[:-1]

        try:
            cursor = connection.cursor()
            if params:
                cursor.execute(sql, params)
            else:
                cursor.execute(sql)

            # Handle different query types
            if sql.strip().upper().startswith(('SELECT', 'SHOW', 'DESC', 'EXPLAIN')):
                rows = cursor.fetchall()
                columns = [desc[0] for desc in cursor.description] if cursor.description else []
                # Convert rows to dictionaries
                data = []
                if rows:
                    for row in rows:
                        row_dict = {}
                        for i, col in enumerate(columns):
                            value = row[i]
                            if isinstance(value, LOB):
                                lob_content = value.read()
                                if hasattr(lob_content, 'decode'):
                                    value = lob_content.decode('utf-8', errors='replace')
                                else:
                                    # For binary LOBs like BLOB, convert to hex or base64; here use str for compatibility
                                    value = str(lob_content)
                            row_dict[col] = value
                        data.append(row_dict)
            else:
                raise Exception("Write operations are not allowed in read-only mode")

            return QueryResult(
                success=True,
                data=data,
                columns=columns,
                row_count=len(data)
            )
        except Exception as e:
            self.logger.error(f"Query failed: {e}\nSQL: {sql}")
            return QueryResult(
                success=False,
                error=str(e)
            )
        finally:
            if cursor:
                cursor.close()

    def get_schema_info(self, connection: oracledb.Connection) -> Dict[str, Any]:
        """Get complete schema information with formatted DDL"""
        schema = {'tables': {}}

        # Get current user/schema
        user_result = self.execute_query(connection, "SELECT USER FROM DUAL")
        if not user_result.success or not user_result.data:
            return schema

        schema_name = user_result.data[0]['USER']
        if not schema_name:
            self.logger.warning("No schema/user found")
            return schema

        # Get all tables owned by the user
        tables_result = self.execute_query(connection,
            "SELECT TABLE_NAME FROM ALL_TABLES WHERE OWNER = :owner AND TABLE_NAME NOT LIKE 'BIN$%'",  # Exclude recycle bin tables
            (schema_name,))

        if not tables_result.success:
            return schema

        for row in tables_result.data:
            table_name = row['TABLE_NAME']
            table_info = self._get_table_info(connection, schema_name, table_name)
            schema['tables'][table_name] = table_info

        schema['db_details'] = self._get_db_details(schema)
        return schema

    def _get_table_info(self, connection: oracledb.Connection, schema_name: str, table_name: str) -> Dict[str, Any]:
        """Get detailed table information"""
        table_info = {
            'columns': {},
            'primary_keys': [],
            'foreign_keys': [],
            'sample_data': [],
            'create_statement': None,
            'insert_statement': []
        }

        # Get columns information
        col_result = self.execute_query(connection, """
            SELECT COLUMN_NAME, DATA_TYPE, DATA_LENGTH, DATA_PRECISION, DATA_SCALE,
                   NULLABLE, DATA_DEFAULT, CHAR_LENGTH
            FROM ALL_TAB_COLUMNS
            WHERE OWNER = :owner AND TABLE_NAME = :table_name
            ORDER BY COLUMN_ID
        """, (schema_name, table_name))

        if col_result.success:
            for col in col_result.data:
                col_name = col['COLUMN_NAME']
                data_type = col['DATA_TYPE']

                # Format data type
                if data_type == 'NUMBER':
                    if col['DATA_PRECISION'] is not None:
                        if col['DATA_SCALE'] is not None and col['DATA_SCALE'] > 0:
                            data_type = f"NUMBER({col['DATA_PRECISION']},{col['DATA_SCALE']})"
                        else:
                            data_type = f"NUMBER({col['DATA_PRECISION']})"
                    else:
                        data_type = "NUMBER"
                elif data_type in ('VARCHAR2', 'NVARCHAR2', 'CHAR', 'NCHAR'):
                    if col['CHAR_LENGTH'] is not None:
                        data_type = f"{data_type}({col['CHAR_LENGTH']})"
                elif data_type in ('DATE', 'TIMESTAMP', 'BLOB', 'CLOB', 'NCLOB'):
                    pass  # No length specification needed
                else:
                    if col['DATA_LENGTH'] is not None:
                        data_type = f"{data_type}({col['DATA_LENGTH']})"

                default = col['DATA_DEFAULT']
                if default is not None:
                    default = default.strip()

                table_info['columns'][col_name] = {
                    'type': data_type,
                    'nullable': col['NULLABLE'] == 'Y',
                    'default': default
                }

        # Get primary keys
        pk_result = self.execute_query(connection, """
            SELECT acc.COLUMN_NAME
            FROM ALL_CONS_COLUMNS acc
            JOIN ALL_CONSTRAINTS ac ON acc.CONSTRAINT_NAME = ac.CONSTRAINT_NAME
                AND acc.OWNER = ac.OWNER
            WHERE ac.OWNER = :owner AND ac.TABLE_NAME = :table_name
                AND ac.CONSTRAINT_TYPE = 'P'
            ORDER BY acc.POSITION
        """, (schema_name, table_name))

        if pk_result.success:
            table_info['primary_keys'] = [row['COLUMN_NAME'] for row in pk_result.data]

        # Mark primary key columns
        for pk in table_info['primary_keys']:
            if pk in table_info['columns']:
                table_info['columns'][pk]['primary_key'] = True

        # Get foreign keys
        fk_result = self.execute_query(connection, """
            SELECT acc.COLUMN_NAME,
                   ac2.TABLE_NAME AS REFERENCED_TABLE_NAME,
                   acc2.COLUMN_NAME AS REFERENCED_COLUMN_NAME,
                   ac.DELETE_RULE,
                   'NO ACTION' AS UPDATE_RULE  -- Oracle doesn't have UPDATE_RULE like MySQL
            FROM ALL_CONS_COLUMNS acc
            JOIN ALL_CONSTRAINTS ac ON acc.CONSTRAINT_NAME = ac.CONSTRAINT_NAME
                AND acc.OWNER = ac.OWNER
            JOIN ALL_CONSTRAINTS ac2 ON ac.R_CONSTRAINT_NAME = ac2.CONSTRAINT_NAME
                AND ac.R_OWNER = ac2.OWNER
            JOIN ALL_CONS_COLUMNS acc2 ON ac2.CONSTRAINT_NAME = acc2.CONSTRAINT_NAME
                AND ac2.OWNER = acc2.OWNER
                AND acc.POSITION = acc2.POSITION
            WHERE ac.OWNER = :owner AND ac.TABLE_NAME = :table_name
                AND ac.CONSTRAINT_TYPE = 'R'
        """, (schema_name, table_name))

        if fk_result.success:
            for fk in fk_result.data:
                table_info['foreign_keys'].append({
                    'column': fk['COLUMN_NAME'],
                    'referenced_table': fk['REFERENCED_TABLE_NAME'],
                    'referenced_column': fk['REFERENCED_COLUMN_NAME'],
                    'update_rule': fk['UPDATE_RULE'],
                    'delete_rule': fk['DELETE_RULE']
                })

        # Get sample data (limited to 2 rows)
        sample_result = self.execute_query(connection,
            f"SELECT * FROM \"{schema_name}\".\"{table_name}\" WHERE ROWNUM <= 2")

        if sample_result.success and sample_result.data:
            table_info['sample_data'] = sample_result.data
            column_names = list(sample_result.data[0].keys())

            # Generate INSERT statements
            for row in sample_result.data:
                values = []
                for val in row.values():
                    if val is None:
                        values.append('NULL')
                    elif isinstance(val, (int, float)):
                        values.append(str(val))
                    elif isinstance(val, bool):
                        values.append('1' if val else '0')
                    elif isinstance(val, LOB):
                        lob_content = val.read()
                        if hasattr(lob_content, 'decode'):
                            escaped_val = lob_content.decode('utf-8', errors='replace').replace("'", "''")
                        else:
                            escaped_val = str(lob_content).replace("'", "''")
                        values.append(f"'{escaped_val}'")
                    else:
                        # Oracle uses single quotes for string literals
                        escaped = str(val).replace("'", "''")
                        values.append(f"'{escaped}'")

                column_parts = [f'"{c}"' for c in column_names]
                column_list = ', '.join(column_parts)
                value_list = ', '.join(values)
                insert_sql = f'INSERT INTO "{schema_name}"."{table_name}" ({column_list}) VALUES ({value_list})'
                table_info['insert_statement'].append(insert_sql)

        # Get create statement using DBMS_METADATA
        create_result = self.execute_query(connection, """
            SELECT DBMS_METADATA.GET_DDL('TABLE', :table_name, :schema_name) AS ddl
            FROM DUAL
            UNION ALL
            SELECT DBMS_METADATA.GET_DDL('INDEX', i.INDEX_NAME, i.OWNER) AS ddl
            FROM ALL_INDEXES i
            WHERE i.TABLE_OWNER = :schema_name AND i.TABLE_NAME = :table_name
                AND i.INDEX_NAME NOT IN (
                    SELECT CONSTRAINT_NAME FROM ALL_CONSTRAINTS
                    WHERE OWNER = :schema_name AND TABLE_NAME = :table_name
                )
        """, (table_name, schema_name, schema_name, table_name,schema_name, table_name))

        if create_result.success and create_result.data:
            ddl_statements = [row['DDL'] for row in create_result.data if row['DDL']]
            table_info['create_statement'] = '\n\n'.join(ddl_statements)

        return table_info

    def _get_db_details(self, schema: Dict[str, Any]) -> str:
        """Generate formatted DDL statements from schema information"""
        db_details = []

        for table_name, table_info in schema['tables'].items():
            # Use original create statement if available
            if table_info['create_statement']:
                db_details.append(table_info['create_statement'])
            else:
                # Build create statement from metadata
                column_defs = []
                for col_name, col_info in table_info['columns'].items():
                    col_def = f"    \"{col_name}\" {col_info['type']}"

                    if not col_info['nullable']:
                        col_def += " NOT NULL"

                    if col_info['default'] is not None:
                        col_def += f" DEFAULT {col_info['default']}"

                    column_defs.append(col_def)

                # Add constraints
                constraints = []

                # Primary keys
                if table_info['primary_keys']:
                    pk_cols = ", ".join(f"\"{pk}\"" for pk in table_info['primary_keys'])
                    constraints.append(f"    PRIMARY KEY ({pk_cols})")

                # Foreign keys
                for fk in table_info['foreign_keys']:
                    fk_def = (
                        f"    CONSTRAINT \"fk_{table_name}_{fk['column']}\" "
                        f"FOREIGN KEY (\"{fk['column']}\") "
                        f"REFERENCES \"{fk['referenced_table']}\" (\"{fk['referenced_column']}\")"
                    )
                    if fk.get('delete_rule'):
                        fk_def += f" ON DELETE {fk['delete_rule']}"
                    constraints.append(fk_def)

                create_table = f"CREATE TABLE \"{table_name}\" (\n"
                create_table += ",\n".join(column_defs + constraints)
                create_table += "\n);"

                db_details.append(create_table)

            # Add sample data if available
            if table_info['sample_data']:
                db_details.append("\n-- Sample data:")
                db_details.extend(table_info['insert_statement'])

        return "\n\n".join(db_details)

    def discover_databases(self, config: Dict) -> Dict[str, DatabaseInfo]:
        """Discover Oracle schemas/databases"""
        databases = {}

        try:
            # Create temp connection
            conn = self.connect(config)

            # Get schema list (excluding system schemas)
           
            #result = self.execute_query(conn, """
            #    SELECT USERNAME AS schema_name
            #    FROM ALL_USERS
            #    WHERE USERNAME NOT IN ('SYS', 'SYSTEM', 'SYSMAN', 'DBSNMP', 'OUTLN',
            #                          'APPQOSSYS', 'DBSFWUSER', 'DIP', 'GSMADMIN_INTERNAL',
            #                          'GSMCATUSER', 'GSMUSER', 'ORACLE_OCM', 'REMOTE_SCHEDULER_AGENT',
            #                          'SYSBACKUP', 'SYSDG', 'SYSKM', 'SYSRAC', 'WMSYS', 'XDB',
            #                          'XS$NULL', 'PERFSTAT', 'EXFSYS', 'CTXSYS', 'ORDSYS',
            #                          'ORDPLUGINS', 'SI_INFORMTN_SCHEMA', 'MDSYS', 'OLAPSYS',
            #                          'APEX_030200', 'APEX_PUBLIC_USER', 'FLOWS_FILES', 'OWBSYS',
            #                          'OWBSYS_AUDIT', 'SCOTT', 'OE', 'PM', 'IX','
            #                          'BI', 'ANONYMOUS', 'AURORA$JIS$UTILITY$', 'AURORA$ORB$UNAUTHENTICATED',
            #                          'DSSYS', 'OJVMSYS', 'LBACSYS', 'DVSYS', 'DVF', 'AUDSYS')
            #    ORDER BY USERNAME
            # )
            user = config['user']
            result = self.execute_query(conn, """
                SELECT USERNAME AS schema_name
                FROM ALL_USERS
                WHERE USERNAME = :1
            """,(user,))

            if result.success:
                for row in result.data:
                    schema_name = row.get('SCHEMA_NAME')
                    if not schema_name:
                        continue
                    databases[schema_name] = DatabaseInfo(
                        db_id=schema_name,
                        db_type='oracle',
                        connection_info={**config, 'user': schema_name},  # Note: user/schema distinction
                        metadata={
                            'host': config.get('host', 'localhost'),
                            'port': config.get('port', 1521),
                            'service_name': config.get('service_name', config.get('database', 'ORCL')),
                            'schema': schema_name
                        }
                    )
                    self.logger.info(f"Discovered Oracle schema: {schema_name}")

            conn.close()
        except Exception as e:
            self.logger.error(f"Schema discovery failed: {e}")

        return databases

    def get_number_of_special_column(self, connection: oracledb.Connection) -> int:
        """
        Get the number of columns ending with '_embedding' in database
        """
        count = 0
        try:
            # Get the complete structure information of the database
            schema = self.get_schema_info(connection)

            # Get all the table information from the schema
            tables = schema.get('tables', {})

            # Traverse each table
            for table_name, table_info in tables.items():
                # Get all the column names from the table information
                columns = table_info.get('columns', {})

                # Traverse the column names
                for column_name in columns.keys():
                    # Check if the column name ends with '_embedding'
                    if isinstance(column_name, str) and column_name.endswith('_embedding'):
                        count += 1

        except Exception as e:
            self.logger.error(f"Error counting embedding columns: {e}")

        if count == 0:
            self.logger.warning("No columns ending with '_embedding' found.")
        return count
