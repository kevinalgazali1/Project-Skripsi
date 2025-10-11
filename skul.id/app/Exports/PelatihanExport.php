<?php

namespace App\Exports;

use Carbon\Carbon;
use App\Models\Pelatihan;
use Illuminate\Support\Facades\Auth;
use Maatwebsite\Excel\Concerns\WithHeadings;
use Illuminate\Contracts\Support\Responsable;
use Maatwebsite\Excel\Concerns\FromCollection;

class PelatihanExport implements FromCollection, WithHeadings
{
    protected $request;

    public function __construct($request)
    {
        $this->request = $request;
    }

    public function collection()
    {
        $user = Auth::user();

        $query = Pelatihan::with(['user.mitraProfile', 'daftarPelatihan'])
            ->withCount('daftarPelatihan')
            ->orderBy('created_at', 'desc');

        // Jika bukan admin, filter berdasarkan user_id
        if ($user->role !== 'admin') {
            $query->where('user_id', $user->id);
        }

        // Terapkan semua filter
        if ($this->request->filled('search')) {
            $query->where('nama_pelatihan', 'like', '%' . $this->request->judul . '%');
        }

        if ($this->request->filled('kota')) {
            $query->where('kota', 'like', '%' . $this->request->kota . '%');
        }

        if ($this->request->filled('status')) {
            $query->where('status', $this->request->status);
        }

        if ($this->request->filled('tahun_mulai')) {
            $query->whereYear('tanggal_mulai', $this->request->tahun_mulai);
        }

        if ($this->request->filled('status_selesai')) {
            if ($this->request->status_selesai == 'selesai') {
                $query->whereDate('tanggal_selesai', '<', Carbon::today());
            } elseif ($this->request->status_selesai == 'belum') {
                $query->whereDate('tanggal_selesai', '>=', Carbon::today());
            }
        }

        $pelatihan = $query->get();

        return $pelatihan->map(function ($p) {
            $pesertas = collect($p->daftarPelatihan);

            return [
                'Instansi' => $p->user->mitraProfile->nama_instansi ?? '-',
                'Nama Pelatihan' => $p->nama_pelatihan ?? '-',
                'Kota' => $p->kota ?? '-',
                'Jumlah Peserta' => $p->daftar_sertifikasis_count ?? 0,
                'Jumlah Asal Sekolah' => $pesertas->pluck('asal_sekolah')->unique()->count() ?? 0,
                'Jumlah Jurusan' => $pesertas->pluck('jurusan')->unique()->count() ?? 0,
                'Jumlah Bekerja/Wirausaha' => $pesertas->whereIn('status_saat_ini', ['Bekerja', 'Wirausaha'])->count() ?? 0,
                'Jumlah Tidak Bekerja' => $pesertas->where('status_saat_ini', 'Tidak Bekerja')->count() ?? 0,
                'Jumlah Kuliah' => $pesertas->where('status_saat_ini', 'Kuliah')->count() ?? 0,
            ];
        });
    }

    public function headings(): array
    {
        return [
            'Instansi',
            'Nama Pelatihan',
            'Kota',
            'Jumlah Peserta',
            'Jumlah Asal Sekolah',
            'Jumlah Jurusan',
            'Jumlah Bekerja/Wirausaha',
            'Jumlah Tidak Bekerja',
            'Jumlah Kuliah',
        ];
    }
}
