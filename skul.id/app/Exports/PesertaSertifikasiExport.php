<?php

namespace App\Exports;

use App\Models\DaftarSertifikasi;
use Maatwebsite\Excel\Concerns\FromCollection;
use Maatwebsite\Excel\Concerns\WithHeadings;

class PesertaSertifikasiExport implements FromCollection, WithHeadings
{
    protected $sertifikasi_id;

    public function __construct($sertifikasi_id)
    {
        $this->sertifikasi_id = $sertifikasi_id;
    }

    public function collection()
    {
        $data = DaftarSertifikasi::where('sertifikasi_id', $this->sertifikasi_id)->get();

        return $data->map(function ($item) {
            return [
                $item->nama_lengkap,
                $item->email,
                $item->no_hp,
                $item->asal_sekolah,
                $item->jurusan,
                $item->jenis_kelamin,
                $item->tanggal_lahir,
                "'" . $item->nik,
                $item->tahun_kelulusan,
                $item->npsn,
                $item->provinsi,
                $item->kota,
                $item->alamat,
                $item->status_saat_ini,
                $item->bidang_pekerjaan,
                $item->sertifikasi_terakhir,
                $item->kesesuaian_sertifikasi,
                $item->nama_universitas,
                $item->jurusan_universitas,
            ];
        });
    }

    public function headings(): array
    {
        return [
            'Nama Lengkap',
            'Email',
            'No. Telepon',
            'Asal Sekolah',
            'Jurusan',
            'Jenis Kelamin',
            'Tanggal Lahir',
            'NIK',
            'Tahun Kelulusan',
            'NPSN',
            'Provinsi',
            'Kota',
            'Alamat',
            'Status Saat Ini',
            'Bidang Pekerjaan',
            'Sertifikasi Terakhir',
            'Kesesuaian Sertifikasi',
            'Nama Universitas',
            'Jurusan Universitas',
        ];
    }
}
