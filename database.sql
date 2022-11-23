create database DetectSignature
go
use DetectSignature
go
create table Customers(
    Id int identity primary key,
    Name nvarchar(100),
    Featured nvarchar(max),
)